"""
JARVIS Communication Agent
Handles all user communication, notifications, voice, and external messaging
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

from app.core.base_agent import BaseAgent, AgentState
from app.core.message_bus import (
    AgentMessage, MessageType, AgentType, MessageBuilder, message_bus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Communication channels"""
    VOICE = "voice"
    TEXT = "text"
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"
    CHAT = "chat"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Notification:
    """A notification to be sent"""
    def __init__(
        self,
        notification_id: str,
        title: str,
        message: str,
        channel: ChannelType,
        priority: MessagePriority = MessagePriority.NORMAL,
        data: Dict = None,
        actions: List[Dict] = None
    ):
        self.notification_id = notification_id
        self.title = title
        self.message = message
        self.channel = channel
        self.priority = priority
        self.data = data or {}
        self.actions = actions or []
        self.created_at = datetime.utcnow().isoformat()
        self.delivered = False
        self.delivered_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "title": self.title,
            "message": self.message,
            "channel": self.channel.value,
            "priority": self.priority.value,
            "data": self.data,
            "actions": self.actions,
            "created_at": self.created_at,
            "delivered": self.delivered
        }


class VoiceSession:
    """An active voice conversation session"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_active = False
        self.is_listening = False
        self.is_speaking = False
        self.language = "en"
        self.context: List[Dict] = []
        self.started_at = datetime.utcnow().isoformat()
        self.last_activity = self.started_at
    
    def add_to_context(self, role: str, content: str):
        """Add to conversation context"""
        self.context.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_activity = datetime.utcnow().isoformat()
        
        # Keep context manageable
        if len(self.context) > 20:
            self.context = self.context[-20:]


class CommunicationAgent(BaseAgent):
    """
    Communication Agent: Manages all communication
    
    Capabilities:
    - Voice conversation management
    - Text-based chat
    - Push notifications
    - Email sending
    - SMS messaging
    - Multi-channel coordination
    """
    
    def __init__(
        self,
        stt_client=None,
        tts_client=None,
        email_client=None,
        config: Dict = None
    ):
        super().__init__(AgentType.COMMUNICATION, config=config)
        
        # Clients
        self.stt_client = stt_client  # Speech-to-Text
        self.tts_client = tts_client  # Text-to-Speech
        self.email_client = email_client
        
        # Voice sessions
        self.voice_sessions: Dict[str, VoiceSession] = {}
        self.active_voice_session: Optional[str] = None
        
        # Notifications
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.pending_notifications: Dict[str, Notification] = {}
        
        # User preferences
        self.user_preferences: Dict[str, Any] = {
            "preferred_channel": ChannelType.TEXT.value,
            "voice_enabled": True,
            "notification_enabled": True,
            "quiet_hours": {"start": 22, "end": 8},
            "language": "en"
        }
        
        # Callbacks for frontend integration
        self.on_voice_output: Optional[Callable] = None
        self.on_notification: Optional[Callable] = None
        self.on_text_response: Optional[Callable] = None
    
    async def _initialize(self) -> bool:
        """Initialize the Communication Agent"""
        logger.info("📢 Communication Agent initializing...")
        
        # Register message handlers
        self.register_handler(MessageType.VOICE_INPUT, self._handle_voice_input)
        self.register_handler(MessageType.VOICE_OUTPUT, self._handle_voice_output)
        self.register_handler(MessageType.REQUEST, self._handle_general_request)
        
        # Start notification processor
        asyncio.create_task(self._notification_processor())
        
        logger.info("✅ Communication Agent initialized")
        return True
    
    async def _cleanup(self):
        """Cleanup resources"""
        # Close all voice sessions
        for session_id in list(self.voice_sessions.keys()):
            await self.end_voice_session(session_id)
    
    async def _handle_message(self, message: AgentMessage):
        """Handle generic messages"""
        logger.debug(f"📥 Communication Agent received: {message.message_type}")
    
    async def _handle_voice_input(self, message: AgentMessage):
        """Handle voice input"""
        content = message.content
        audio_data = content.get("audio_data")
        session_id = content.get("session_id")
        language = content.get("language", "en")
        
        logger.info(f"🎤 Voice input received")
        
        # Get or create voice session
        session = self.voice_sessions.get(session_id)
        if not session:
            session = await self.start_voice_session(language)
            session_id = session.session_id
        
        # Transcribe audio
        if self.stt_client and audio_data:
            try:
                transcript = await self.stt_client.transcribe(audio_data, language)
                
                # Add to session context
                session.add_to_context("user", transcript)
                
                # Send response
                await self.send_response(
                    original_message=message,
                    content={
                        "session_id": session_id,
                        "transcript": transcript,
                        "context": session.context
                    }
                )
                
                # Broadcast for other agents
                await self.broadcast(
                    message_type=MessageType.VOICE_INPUT,
                    content={
                        "session_id": session_id,
                        "transcript": transcript,
                        "source": "communication_agent"
                    }
                )
                
            except Exception as e:
                logger.error(f"❌ STT error: {e}")
                await self.send_error(message.from_agent, str(e), message.message_id)
    
    async def _handle_voice_output(self, message: AgentMessage):
        """Handle voice output requests"""
        content = message.content
        text = content.get("text", "")
        session_id = content.get("session_id")
        voice_id = content.get("voice_id")
        
        logger.info(f"🔊 Voice output: {text[:50]}...")
        
        # Get session
        session = self.voice_sessions.get(session_id) if session_id else None
        
        # Generate speech
        if self.tts_client:
            try:
                audio_data = await self.tts_client.synthesize(text, voice_id)
                
                # Add to session context
                if session:
                    session.add_to_context("assistant", text)
                
                # Send to frontend via callback
                if self.on_voice_output:
                    await self.on_voice_output({
                        "session_id": session_id,
                        "audio_data": audio_data,
                        "text": text
                    })
                
                # Send response
                await self.send_response(
                    original_message=message,
                    content={
                        "session_id": session_id,
                        "text": text,
                        "audio_generated": True
                    }
                )
                
            except Exception as e:
                logger.error(f"❌ TTS error: {e}")
                await self.send_error(message.from_agent, str(e), message.message_id)
    
    async def _handle_general_request(self, message: AgentMessage):
        """Handle general communication requests"""
        content = message.content
        request_type = content.get("type", "")
        
        if request_type == "send_email":
            await self._handle_email_request(message)
        elif request_type == "notification":
            await self._handle_notification_request(message)
        elif request_type == "send_message":
            await self._handle_message_request(message)
        elif request_type == "permission_request":
            await self._handle_permission_notification(message)
    
    async def _handle_email_request(self, message: AgentMessage):
        """Handle email sending requests"""
        content = message.content
        
        to = content.get("to", "")
        subject = content.get("subject", "")
        body = content.get("body", "")
        
        logger.info(f"📧 Email request: {subject} → {to}")
        
        if self.email_client:
            try:
                result = await self.email_client.send(to, subject, body)
                await self.send_response(
                    original_message=message,
                    content={"sent": True, "message_id": result}
                )
            except Exception as e:
                await self.send_response(
                    original_message=message,
                    content={"sent": False, "error": str(e)},
                    success=False
                )
        else:
            await self.send_response(
                original_message=message,
                content={"sent": False, "error": "Email client not configured"},
                success=False
            )
    
    async def _handle_notification_request(self, message: AgentMessage):
        """Handle notification requests"""
        content = message.content
        
        notification = Notification(
            notification_id=f"notif_{datetime.utcnow().timestamp()}",
            title=content.get("title", "Notification"),
            message=content.get("message", ""),
            channel=ChannelType(content.get("channel", "push")),
            priority=MessagePriority(content.get("priority", "normal")),
            data=content.get("data", {}),
            actions=content.get("actions", [])
        )
        
        await self.queue_notification(notification)
        
        await self.send_response(
            original_message=message,
            content={"notification_id": notification.notification_id, "queued": True}
        )
    
    async def _handle_message_request(self, message: AgentMessage):
        """Handle general message requests"""
        content = message.content
        channel = content.get("channel", "text")
        text = content.get("text", "")
        
        logger.info(f"💬 Message via {channel}: {text[:50]}...")
        
        # Send via callback
        if self.on_text_response:
            await self.on_text_response({
                "channel": channel,
                "text": text,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        await self.send_response(
            original_message=message,
            content={"delivered": True}
        )
    
    async def _handle_permission_notification(self, message: AgentMessage):
        """Handle permission request notifications"""
        content = message.content
        permission = content.get("permission", {})
        urgent = content.get("urgent", False)
        
        # Create high-priority notification
        notification = Notification(
            notification_id=f"perm_notif_{permission.get('request_id')}",
            title="Permission Required",
            message=f"Action '{permission.get('action')}' requires your approval",
            channel=ChannelType.PUSH,
            priority=MessagePriority.URGENT if urgent else MessagePriority.HIGH,
            data={"permission_request": permission},
            actions=[
                {"id": "approve", "label": "Approve"},
                {"id": "deny", "label": "Deny"}
            ]
        )
        
        await self.queue_notification(notification)
    
    # Voice session management
    async def start_voice_session(self, language: str = "en") -> VoiceSession:
        """Start a new voice session"""
        session_id = f"voice_{datetime.utcnow().timestamp()}"
        session = VoiceSession(session_id)
        session.language = language
        session.is_active = True
        
        self.voice_sessions[session_id] = session
        self.active_voice_session = session_id
        
        logger.info(f"🎙️ Voice session started: {session_id}")
        
        # Send greeting
        greeting = self._get_greeting(language)
        await self.speak(greeting, session_id)
        
        return session
    
    async def end_voice_session(self, session_id: str):
        """End a voice session"""
        session = self.voice_sessions.get(session_id)
        if session:
            session.is_active = False
            del self.voice_sessions[session_id]
            
            if self.active_voice_session == session_id:
                self.active_voice_session = None
            
            logger.info(f"🎙️ Voice session ended: {session_id}")
    
    async def speak(self, text: str, session_id: str = None, voice_id: str = None):
        """Speak text via voice"""
        if not self.tts_client:
            logger.warning("TTS client not available")
            return
        
        # Use active session if none specified
        if not session_id and self.active_voice_session:
            session_id = self.active_voice_session
        
        # Send voice output message
        await self.send_message(
            to_agent=AgentType.COMMUNICATION,  # Self
            message_type=MessageType.VOICE_OUTPUT,
            content={
                "text": text,
                "session_id": session_id,
                "voice_id": voice_id
            }
        )
    
    async def send_text(
        self,
        text: str,
        channel: ChannelType = ChannelType.TEXT,
        priority: MessagePriority = MessagePriority.NORMAL
    ):
        """Send text message"""
        # Send via callback
        if self.on_text_response:
            await self.on_text_response({
                "channel": channel.value,
                "text": text,
                "priority": priority.value,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    # Notification management
    async def queue_notification(self, notification: Notification):
        """Queue a notification for delivery"""
        await self.notification_queue.put(notification)
        self.pending_notifications[notification.notification_id] = notification
        logger.info(f"📬 Notification queued: {notification.title}")
    
    async def _notification_processor(self):
        """Background notification processor"""
        while self.running:
            try:
                notification = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )
                
                await self._deliver_notification(notification)
                
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"❌ Notification delivery error: {e}")
    
    async def _deliver_notification(self, notification: Notification):
        """Deliver a notification"""
        # Check quiet hours for non-urgent notifications
        if notification.priority != MessagePriority.URGENT:
            if self._is_quiet_hours():
                logger.info("⏸️ Notification deferred due to quiet hours")
                return
        
        # Deliver via appropriate channel
        if notification.channel == ChannelType.PUSH:
            await self._send_push_notification(notification)
        elif notification.channel == ChannelType.EMAIL:
            await self._send_email_notification(notification)
        elif notification.channel == ChannelType.SMS:
            await self._send_sms_notification(notification)
        
        # Mark as delivered
        notification.delivered = True
        notification.delivered_at = datetime.utcnow().isoformat()
        
        # Notify via callback
        if self.on_notification:
            await self.on_notification(notification.to_dict())
        
        logger.info(f"📬 Notification delivered: {notification.title}")
    
    async def _send_push_notification(self, notification: Notification):
        """Send push notification"""
        # This would integrate with FCM/APNs
        logger.info(f"📱 Push: {notification.title}")
    
    async def _send_email_notification(self, notification: Notification):
        """Send email notification"""
        if self.email_client:
            await self.email_client.send(
                to="user@example.com",  # Would come from user profile
                subject=notification.title,
                body=notification.message
            )
    
    async def _send_sms_notification(self, notification: Notification):
        """Send SMS notification"""
        # This would integrate with SMS gateway
        logger.info(f"📱 SMS: {notification.title}")
    
    def _is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours"""
        now = datetime.utcnow().hour
        quiet_start = self.user_preferences["quiet_hours"]["start"]
        quiet_end = self.user_preferences["quiet_hours"]["end"]
        
        if quiet_start <= quiet_end:
            return quiet_start <= now < quiet_end
        else:
            return now >= quiet_start or now < quiet_end
    
    def _get_greeting(self, language: str) -> str:
        """Get appropriate greeting based on time"""
        hour = datetime.utcnow().hour
        
        greetings = {
            "en": {
                "morning": "Good morning! I'm JARVIS. How can I help you today?",
                "afternoon": "Good afternoon! I'm JARVIS. What can I do for you?",
                "evening": "Good evening! I'm JARVIS. How may I assist you?"
            },
            "hi": {
                "morning": "शुभ प्रभात! मैं JARVIS हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
                "afternoon": "नमस्ते! मैं JARVIS हूँ। मैं आपके लिए क्या कर सकता हूँ?",
                "evening": "शुभ संध्या! मैं JARVIS हूँ। मैं आपकी सहायता कैसे कर सकता हूँ?"
            }
        }
        
        lang_greetings = greetings.get(language, greetings["en"])
        
        if 5 <= hour < 12:
            return lang_greetings["morning"]
        elif 12 <= hour < 17:
            return lang_greetings["afternoon"]
        else:
            return lang_greetings["evening"]
    
    # User preference management
    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user communication preferences"""
        self.user_preferences.update(preferences)
        logger.info("👤 User preferences updated")
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get user communication preferences"""
        return self.user_preferences
    
    # Stats
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "active_voice_sessions": len(self.voice_sessions),
            "pending_notifications": len(self.pending_notifications),
            "notification_queue_size": self.notification_queue.qsize(),
            "user_preferences": self.user_preferences
        }
