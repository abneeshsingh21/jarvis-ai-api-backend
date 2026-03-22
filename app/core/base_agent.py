"""
JARVIS Base Agent - Foundation for all 6 specialized agents
Provides common functionality for message handling, lifecycle management
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import traceback

from app.core.message_bus import (
    MessageBus, AgentMessage, MessageType, AgentType, MessageBuilder, message_bus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState:
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class BaseAgent(ABC):
    """
    Base class for all JARVIS agents
    Handles message processing, state management, and lifecycle
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        msg_bus: MessageBus = None,
        config: Dict[str, Any] = None
    ):
        self.agent_type = agent_type
        self.agent_name = agent_type.value
        self.message_bus = msg_bus or message_bus
        self.config = config or {}
        
        # State management
        self.state = AgentState.INITIALIZING
        self.state_history: List[Dict[str, Any]] = []
        
        # Message handling
        self.message_queue: Optional[asyncio.Queue] = None
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "messages_processed": 0,
            "messages_sent": 0,
            "errors": 0,
            "start_time": None,
            "tasks_completed": 0
        }
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[MessageType.HEARTBEAT.value] = self._handle_heartbeat
        self.message_handlers[MessageType.STATUS_UPDATE.value] = self._handle_status_update
        self.message_handlers[MessageType.ERROR.value] = self._handle_error_message
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            logger.info(f"🚀 Initializing {self.agent_name} agent...")
            self._set_state(AgentState.INITIALIZING)
            
            # Register with message bus
            self.message_queue = self.message_bus.register_agent(self.agent_type)
            
            # Agent-specific initialization
            success = await self._initialize()
            
            if success:
                self._set_state(AgentState.IDLE)
                logger.info(f"✅ {self.agent_name} agent initialized")
            else:
                self._set_state(AgentState.ERROR)
                logger.error(f"❌ {self.agent_name} agent initialization failed")
            
            return success
            
        except Exception as e:
            self._set_state(AgentState.ERROR)
            logger.error(f"❌ {self.agent_name} initialization error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    @abstractmethod
    async def _initialize(self) -> bool:
        """Agent-specific initialization - override in subclass"""
        return True
    
    async def start(self):
        """Start the agent's message processing loop"""
        if self.running:
            logger.warning(f"{self.agent_name} already running")
            return
        
        self.running = True
        self.metrics["start_time"] = datetime.utcnow().isoformat()
        self._task = asyncio.create_task(self._message_loop())
        logger.info(f"▶️ {self.agent_name} agent started")
    
    async def stop(self):
        """Stop the agent"""
        if not self.running:
            return
        
        self.running = False
        self._set_state(AgentState.SHUTDOWN)
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Unregister from message bus
        self.message_bus.unregister_agent(self.agent_type)
        
        # Agent-specific cleanup
        await self._cleanup()
        
        logger.info(f"⏹️ {self.agent_name} agent stopped")
    
    @abstractmethod
    async def _cleanup(self):
        """Agent-specific cleanup - override in subclass"""
        pass
    
    async def _message_loop(self):
        """Main message processing loop"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = await self.message_bus.get_message(self.agent_type, timeout=1.0)
                
                if message:
                    await self._process_message(message)
                
                # Periodic tasks
                await self._periodic_tasks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ {self.agent_name} message loop error: {e}")
                self.metrics["errors"] += 1
                await asyncio.sleep(1)
    
    async def _process_message(self, message: AgentMessage):
        """Process incoming message"""
        try:
            logger.debug(f"📥 {self.agent_name} received: {message.message_type}")
            
            # Update state based on message type
            await self._update_state_for_message(message)
            
            # Check for specific handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                # Use default processing
                await self._handle_message(message)
            
            self.metrics["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            logger.error(traceback.format_exc())
            self.metrics["errors"] += 1
            
            # Send error response
            await self.send_error(message.from_agent, str(e), message.message_id)
        finally:
            # Return to idle state
            if self.state != AgentState.ERROR:
                self._set_state(AgentState.IDLE)
    
    async def _update_state_for_message(self, message: AgentMessage):
        """Update agent state based on message type"""
        state_map = {
            MessageType.THINK.value: AgentState.THINKING,
            MessageType.PLAN.value: AgentState.PLANNING,
            MessageType.EXECUTE.value: AgentState.EXECUTING,
            MessageType.REFLECT.value: AgentState.REFLECTING,
        }
        
        new_state = state_map.get(message.message_type)
        if new_state:
            self._set_state(new_state)
    
    @abstractmethod
    async def _handle_message(self, message: AgentMessage):
        """Handle messages without specific handlers - override in subclass"""
        pass
    
    # Default message handlers
    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle heartbeat messages"""
        logger.debug(f"💓 Heartbeat from {message.from_agent}")
    
    async def _handle_status_update(self, message: AgentMessage):
        """Handle status update messages"""
        logger.debug(f"📊 Status from {message.from_agent}: {message.content}")
    
    async def _handle_error_message(self, message: AgentMessage):
        """Handle error messages"""
        logger.error(f"⚠️ Error from {message.from_agent}: {message.content}")
    
    async def _periodic_tasks(self):
        """Periodic tasks - override in subclass"""
        pass
    
    # Message sending helpers
    async def send_message(
        self,
        to_agent: AgentType,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 5,
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Send a message to another agent"""
        message = MessageBuilder.create(
            from_agent=self.agent_type,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            priority=priority,
            correlation_id=correlation_id
        )
        
        await self.message_bus.publish(message)
        self.metrics["messages_sent"] += 1
        return message
    
    async def broadcast(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 5
    ) -> AgentMessage:
        """Broadcast a message to all agents"""
        message = MessageBuilder.create(
            from_agent=self.agent_type,
            to_agent=None,  # broadcast
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        await self.message_bus.publish(message)
        self.metrics["messages_sent"] += 1
        return message
    
    async def send_response(
        self,
        original_message: AgentMessage,
        content: Dict[str, Any],
        success: bool = True
    ) -> AgentMessage:
        """Send a response to a message"""
        response_content = {
            "success": success,
            "data": content,
            "in_response_to": original_message.message_id
        }
        
        return await self.send_message(
            to_agent=AgentType(original_message.from_agent),
            message_type=MessageType.RESPONSE,
            content=response_content,
            priority=2,
            correlation_id=original_message.correlation_id
        )
    
    async def send_error(
        self,
        to_agent: str,
        error_message: str,
        correlation_id: Optional[str] = None
    ):
        """Send an error message"""
        await self.send_message(
            to_agent=AgentType(to_agent) if to_agent != "broadcast" else None,
            message_type=MessageType.ERROR,
            content={"error": error_message},
            priority=1,
            correlation_id=correlation_id
        )
    
    # State management
    def _set_state(self, new_state: str):
        """Update agent state"""
        old_state = self.state
        self.state = new_state
        self.state_history.append({
            "from": old_state,
            "to": new_state,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_state(self) -> str:
        """Get current state"""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            **self.metrics,
            "current_state": self.state,
            "state_history_count": len(self.state_history)
        }
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type.value] = handler
    
    async def request_memory(
        self,
        query: str,
        context: Dict = None,
        limit: int = 5
    ) -> Optional[List[Dict]]:
        """Request memory retrieval from Memory Agent"""
        message = await self.send_message(
            to_agent=AgentType.MEMORY,
            message_type=MessageType.MEMORY_RETRIEVE,
            content={"query": query, "context": context, "limit": limit},
            priority=2
        )
        
        # Wait for response
        response = await self.message_bus.wait_for_response(
            correlation_id=message.message_id,
            timeout=10.0
        )
        
        if response and response.content.get("success"):
            return response.content.get("data", {}).get("memories", [])
        return None
    
    async def store_memory(
        self,
        key: str,
        data: Any,
        importance: int = 5,
        metadata: Dict = None
    ) -> bool:
        """Store data in memory via Memory Agent"""
        message = await self.send_message(
            to_agent=AgentType.MEMORY,
            message_type=MessageType.MEMORY_STORE,
            content={
                "key": key,
                "data": data,
                "importance": importance,
                "metadata": metadata or {}
            },
            priority=4
        )
        
        # Wait for confirmation
        response = await self.message_bus.wait_for_response(
            correlation_id=message.message_id,
            timeout=5.0
        )
        
        return response is not None and response.content.get("success", False)
