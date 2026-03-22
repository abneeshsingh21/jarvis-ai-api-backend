"""
JARVIS Voice System - Siri-level Voice Assistant
Supports streaming, interrupt handling, Hinglish, and multilingual conversations
"""

import asyncio
import io
import logging
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from datetime import datetime
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceState(Enum):
    """Voice session states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class LanguageMode(Enum):
    """Language modes for voice"""
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"  # Hindi + English mixed
    AUTO_DETECT = "auto"


class StreamingBuffer:
    """Buffer for streaming audio data"""
    
    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size
        self.buffer = io.BytesIO()
        self.lock = asyncio.Lock()
    
    async def write(self, data: bytes):
        """Write data to buffer"""
        async with self.lock:
            self.buffer.write(data)
    
    async def read_chunk(self) -> Optional[bytes]:
        """Read a chunk from buffer"""
        async with self.lock:
            self.buffer.seek(0)
            chunk = self.buffer.read(self.chunk_size)
            remaining = self.buffer.read()
            
            self.buffer = io.BytesIO()
            self.buffer.write(remaining)
            
            return chunk if chunk else None
    
    async def clear(self):
        """Clear the buffer"""
        async with self.lock:
            self.buffer = io.BytesIO()
    
    def size(self) -> int:
        """Get buffer size"""
        return self.buffer.tell()


class STTProvider:
    """Speech-to-Text provider interface"""
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        streaming: bool = False
    ) -> str:
        """Transcribe audio to text"""
        raise NotImplementedError
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "en"
    ) -> AsyncGenerator[str, None]:
        """Stream transcription results"""
        raise NotImplementedError


class TTSProvider:
    """Text-to-Speech provider interface"""
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en"
    ) -> bytes:
        """Synthesize text to audio"""
        raise NotImplementedError
    
    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio chunks"""
        raise NotImplementedError


class WhisperSTT(STTProvider):
    """OpenAI Whisper-based STT"""
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model
        self.client = None
    
    async def initialize(self):
        """Initialize Whisper client"""
        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            logger.error("OpenAI package not installed")
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        streaming: bool = False
    ) -> str:
        """Transcribe audio using Whisper"""
        if not self.client:
            await self.initialize()
        
        try:
            import openai
            
            # Detect Hinglish
            if language == "hinglish":
                language = "hi"  # Whisper uses 'hi' for Hindi
            
            response = await asyncio.to_thread(
                self.client.audio.transcriptions.create,
                model=self.model,
                file=("audio.wav", audio_data),
                language=language if language != "auto" else None,
                response_format="text"
            )
            
            transcript = response if isinstance(response, str) else response.text
            
            # Post-process for Hinglish
            if language == "hi":
                transcript = self._post_process_hinglish(transcript)
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Whisper STT error: {e}")
            return ""
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "en"
    ) -> AsyncGenerator[str, None]:
        """Stream transcription (simulated with chunked processing)"""
        buffer = StreamingBuffer()
        
        async for chunk in audio_stream:
            await buffer.write(chunk)
            
            # Process when buffer is large enough
            if buffer.size() >= 16000:  # ~1 second of audio at 16kHz
                audio_data = await buffer.read_chunk()
                if audio_data:
                    transcript = await self.transcribe(audio_data, language)
                    if transcript:
                        yield transcript
    
    def _post_process_hinglish(self, text: str) -> str:
        """Post-process Hindi transcription for Hinglish output"""
        # This is a simplified version
        # In production, you'd use a proper transliteration library
        return text


class OpenAITTS(TTSProvider):
    """OpenAI TTS provider"""
    
    def __init__(self, api_key: str, model: str = "tts-1"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self.voices = {
            "en": "alloy",
            "hi": "nova",  # Use a warm voice for Hindi
            "hinglish": "nova"
        }
    
    async def initialize(self):
        """Initialize TTS client"""
        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            logger.error("OpenAI package not installed")
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en"
    ) -> bytes:
        """Synthesize text to speech"""
        if not self.client:
            await self.initialize()
        
        try:
            import openai
            
            voice = voice_id or self.voices.get(language, "alloy")
            
            response = await asyncio.to_thread(
                self.client.audio.speech.create,
                model=self.model,
                voice=voice,
                input=text
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"❌ OpenAI TTS error: {e}")
            return b""
    
    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio"""
        # OpenAI TTS doesn't support true streaming
        # We simulate it by yielding the full audio
        audio = await self.synthesize(text, voice_id, language)
        
        # Yield in chunks
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Small delay for streaming effect


class VoiceSession:
    """A voice conversation session"""
    
    def __init__(
        self,
        session_id: str,
        stt: STTProvider,
        tts: TTSProvider,
        language: str = "en"
    ):
        self.session_id = session_id
        self.stt = stt
        self.tts = tts
        self.language = language
        
        self.state = VoiceState.IDLE
        self.context: List[Dict[str, str]] = []
        self.max_context = 10
        
        # Audio buffers
        self.input_buffer = StreamingBuffer()
        self.output_buffer = StreamingBuffer()
        
        # Interrupt handling
        self.interrupt_event = asyncio.Event()
        self.is_speaking = False
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_transcript: Optional[Callable] = None
        self.on_audio_output: Optional[Callable] = None
        
        # Session timing
        self.created_at = datetime.utcnow().isoformat()
        self.last_activity = self.created_at
    
    async def start_listening(self):
        """Start listening for user input"""
        self.state = VoiceState.LISTENING
        await self._notify_state_change()
        logger.info(f"🎤 Session {self.session_id}: Listening")
    
    async def process_audio(self, audio_data: bytes) -> str:
        """Process incoming audio"""
        self.state = VoiceState.PROCESSING
        await self._notify_state_change()
        
        try:
            # Transcribe
            transcript = await self.stt.transcribe(audio_data, self.language)
            
            if transcript:
                # Add to context
                self._add_to_context("user", transcript)
                
                # Notify
                if self.on_transcript:
                    await self.on_transcript(transcript, "user")
                
                logger.info(f"📝 Transcript: {transcript}")
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            self.state = VoiceState.ERROR
            await self._notify_state_change()
            return ""
    
    async def speak(
        self,
        text: str,
        interruptible: bool = True
    ) -> bytes:
        """Speak text to user"""
        self.state = VoiceState.SPEAKING
        self.is_speaking = True
        await self._notify_state_change()
        
        try:
            # Check for interrupt before starting
            if interruptible and self.interrupt_event.is_set():
                logger.info("🛑 Speech interrupted before start")
                self.is_speaking = False
                return b""
            
            # Synthesize
            audio = await self.tts.synthesize(text, language=self.language)
            
            # Stream audio with interrupt checking
            if interruptible:
                async for chunk in self.tts.synthesize_stream(text, language=self.language):
                    if self.interrupt_event.is_set():
                        logger.info("🛑 Speech interrupted")
                        self.interrupt_event.clear()
                        break
                    
                    if self.on_audio_output:
                        await self.on_audio_output(chunk)
            else:
                if self.on_audio_output:
                    await self.on_audio_output(audio)
            
            # Add to context
            self._add_to_context("assistant", text)
            
            return audio
            
        except Exception as e:
            logger.error(f"❌ Speech error: {e}")
            return b""
        finally:
            self.is_speaking = False
            self.state = VoiceState.IDLE
            await self._notify_state_change()
    
    async def interrupt(self):
        """Interrupt current speech"""
        if self.is_speaking:
            self.interrupt_event.set()
            self.state = VoiceState.INTERRUPTED
            await self._notify_state_change()
            logger.info("🛑 Interrupted")
    
    def _add_to_context(self, role: str, content: str):
        """Add to conversation context"""
        self.context.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Trim context
        if len(self.context) > self.max_context:
            self.context = self.context[-self.max_context:]
        
        self.last_activity = datetime.utcnow().isoformat()
    
    async def _notify_state_change(self):
        """Notify state change"""
        if self.on_state_change:
            await self.on_state_change(self.state)
    
    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation context"""
        return self.context
    
    def clear_context(self):
        """Clear conversation context"""
        self.context = []


class VoiceSystem:
    """
    JARVIS Voice System - Main controller for voice functionality
    
    Features:
    - Multiple concurrent voice sessions
    - Streaming audio processing
    - Interrupt handling
    - Hinglish support
    - Language detection
    """
    
    def __init__(
        self,
        stt_provider: STTProvider = None,
        tts_provider: TTSProvider = None,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.stt = stt_provider
        self.tts = tts_provider
        
        # Sessions
        self.sessions: Dict[str, VoiceSession] = {}
        self.active_session_id: Optional[str] = None
        
        # Default language
        self.default_language = config.get("default_language", "en") if config else "en"
        
        # Callbacks
        self.on_session_created: Optional[Callable] = None
        self.on_session_ended: Optional[Callable] = None
    
    async def initialize(self):
        """Initialize voice system"""
        logger.info("🎙️ Initializing Voice System...")
        
        if self.stt:
            await self.stt.initialize()
        if self.tts:
            await self.tts.initialize()
        
        logger.info("✅ Voice System initialized")
    
    async def create_session(
        self,
        language: str = None,
        session_id: str = None
    ) -> VoiceSession:
        """Create a new voice session"""
        session_id = session_id or f"voice_{datetime.utcnow().timestamp()}"
        language = language or self.default_language
        
        session = VoiceSession(
            session_id=session_id,
            stt=self.stt,
            tts=self.tts,
            language=language
        )
        
        self.sessions[session_id] = session
        self.active_session_id = session_id
        
        if self.on_session_created:
            await self.on_session_created(session)
        
        logger.info(f"🎙️ Voice session created: {session_id} (lang: {language})")
        
        return session
    
    async def end_session(self, session_id: str = None):
        """End a voice session"""
        session_id = session_id or self.active_session_id
        
        session = self.sessions.get(session_id)
        if session:
            del self.sessions[session_id]
            
            if self.active_session_id == session_id:
                self.active_session_id = None
            
            if self.on_session_ended:
                await self.on_session_ended(session)
            
            logger.info(f"🎙️ Voice session ended: {session_id}")
    
    async def process_audio_chunk(
        self,
        audio_data: bytes,
        session_id: str = None
    ) -> str:
        """Process an audio chunk"""
        session = self._get_session(session_id)
        if not session:
            return ""
        
        return await session.process_audio(audio_data)
    
    async def speak(
        self,
        text: str,
        session_id: str = None,
        interruptible: bool = True
    ):
        """Speak text in a session"""
        session = self._get_session(session_id)
        if not session:
            return
        
        await session.speak(text, interruptible)
    
    async def interrupt(self, session_id: str = None):
        """Interrupt current speech"""
        session = self._get_session(session_id)
        if session:
            await session.interrupt()
    
    def _get_session(self, session_id: str = None) -> Optional[VoiceSession]:
        """Get a session by ID"""
        session_id = session_id or self.active_session_id
        return self.sessions.get(session_id)
    
    def detect_language(self, text: str) -> str:
        """Detect language of text (simplified)"""
        # Check for Hindi characters
        hindi_range = range(0x0900, 0x097F)
        has_hindi = any(ord(c) in hindi_range for c in text)
        
        # Check for Hinglish patterns (Roman Hindi)
        hinglish_patterns = ["hai", "nahi", "kya", "kaise", "main", "aap"]
        has_hinglish = any(p in text.lower() for p in hinglish_patterns)
        
        if has_hindi:
            return "hi"
        elif has_hinglish:
            return "hinglish"
        else:
            return "en"
    
    def get_active_session(self) -> Optional[VoiceSession]:
        """Get the currently active session"""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
    
    def get_all_sessions(self) -> List[VoiceSession]:
        """Get all active sessions"""
        return list(self.sessions.values())
    
    async def cleanup_inactive_sessions(self, max_idle_minutes: int = 30):
        """Clean up inactive sessions"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity)
            if now - last_activity > timedelta(minutes=max_idle_minutes):
                to_remove.append(session_id)
        
        for session_id in to_remove:
            await self.end_session(session_id)


# Factory functions for creating voice system with different providers

def create_voice_system_openai(openai_api_key: str, config: Dict = None) -> VoiceSystem:
    """Create voice system with OpenAI providers"""
    stt = WhisperSTT(api_key=openai_api_key)
    tts = OpenAITTS(api_key=openai_api_key)
    
    return VoiceSystem(stt_provider=stt, tts_provider=tts, config=config)


def create_voice_system_groq(groq_api_key: str, config: Dict = None) -> VoiceSystem:
    """Create voice system with Groq (for STT) + OpenAI (for TTS)"""
    # Groq uses Whisper for STT
    stt = WhisperSTT(api_key=groq_api_key, model="whisper-large-v3")
    
    # For TTS, you'd need another provider or OpenAI
    # This is a placeholder
    tts = OpenAITTS(api_key=config.get("openai_api_key") if config else "")
    
    return VoiceSystem(stt_provider=stt, tts_provider=tts, config=config)
