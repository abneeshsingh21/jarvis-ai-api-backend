"""
JARVIS Message Bus - Central Communication System for Multi-Agent Architecture
Enables JSON-based structured communication between all 6 agents
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for agent communication"""
    # Core communication
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DIRECT = "direct"
    
    # Reasoning loop
    THINK = "think"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    IMPROVE = "improve"
    
    # Task management
    TASK_ASSIGN = "task_assign"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    TASK_PROGRESS = "task_progress"
    
    # Memory operations
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_UPDATE = "memory_update"
    
    # Automation
    AUTOMATION_TRIGGER = "automation_trigger"
    AUTOMATION_RESULT = "automation_result"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # Voice
    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    VOICE_INTERRUPT = "voice_interrupt"
    
    # System
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    STATUS_UPDATE = "status_update"


class AgentType(Enum):
    """The 6 core agents in the JARVIS system"""
    PLANNER = "planner"
    DECISION = "decision"
    EXECUTION = "execution"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentMessage:
    """Standard message format for all agent communication"""
    message_id: str
    from_agent: str
    to_agent: str  # "broadcast" or specific agent name
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    priority: int = 5  # 1-10, lower is higher priority
    correlation_id: Optional[str] = None  # For tracking conversation threads
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class MessageBus:
    """
    Central message bus for agent-to-agent communication
    Implements pub/sub pattern with priority queues
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
        self._lock = asyncio.Lock()
        self._running = False
        
    async def start(self):
        """Start the message bus"""
        self._running = True
        logger.info("🚌 Message Bus started")
        
    async def stop(self):
        """Stop the message bus"""
        self._running = False
        logger.info("🚌 Message Bus stopped")
    
    def register_agent(self, agent_type: AgentType) -> asyncio.Queue:
        """Register an agent and create its message queue"""
        agent_name = agent_type.value
        if agent_name not in self.agent_queues:
            self.agent_queues[agent_name] = asyncio.PriorityQueue()
            logger.info(f"✅ Agent registered: {agent_name}")
        return self.agent_queues[agent_name]
    
    def unregister_agent(self, agent_type: AgentType):
        """Unregister an agent"""
        agent_name = agent_type.value
        if agent_name in self.agent_queues:
            del self.agent_queues[agent_name]
            logger.info(f"❌ Agent unregistered: {agent_name}")
    
    async def publish(self, message: AgentMessage) -> bool:
        """
        Publish a message to the bus
        Returns True if message was delivered to at least one subscriber
        """
        if not self._running:
            logger.warning("Message bus not running")
            return False
        
        async with self._lock:
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history.pop(0)
        
        delivered = False
        
        # Direct message to specific agent
        if message.to_agent != "broadcast" and message.to_agent in self.agent_queues:
            queue = self.agent_queues[message.to_agent]
            await queue.put((message.priority, message))
            delivered = True
            logger.debug(f"📨 Direct message: {message.from_agent} → {message.to_agent}")
        
        # Broadcast to all agents
        elif message.to_agent == "broadcast":
            for agent_name, queue in self.agent_queues.items():
                if agent_name != message.from_agent:  # Don't send back to sender
                    await queue.put((message.priority, message))
                    delivered = True
            logger.debug(f"📢 Broadcast: {message.from_agent} → all agents")
        
        # Notify subscribers
        if message.message_type in self.subscribers:
            for callback in self.subscribers[message.message_type]:
                try:
                    asyncio.create_task(callback(message))
                    delivered = True
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")
        
        return delivered
    
    async def subscribe(self, message_type: MessageType, callback: Callable):
        """Subscribe to a specific message type"""
        msg_type = message_type.value
        if msg_type not in self.subscribers:
            self.subscribers[msg_type] = []
        self.subscribers[msg_type].append(callback)
        logger.info(f"📬 Subscribed to: {msg_type}")
    
    async def unsubscribe(self, message_type: MessageType, callback: Callable):
        """Unsubscribe from a message type"""
        msg_type = message_type.value
        if msg_type in self.subscribers and callback in self.subscribers[msg_type]:
            self.subscribers[msg_type].remove(callback)
    
    async def get_message(self, agent_type: AgentType, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Get next message for an agent"""
        agent_name = agent_type.value
        if agent_name not in self.agent_queues:
            return None
        
        queue = self.agent_queues[agent_name]
        try:
            if timeout:
                priority, message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                priority, message = await queue.get()
            return message
        except asyncio.TimeoutError:
            return None
    
    async def peek_messages(self, agent_type: AgentType) -> List[AgentMessage]:
        """Peek at pending messages without removing them"""
        agent_name = agent_type.value
        if agent_name not in self.agent_queues:
            return []
        
        queue = self.agent_queues[agent_name]
        messages = []
        temp_items = []
        
        while not queue.empty():
            priority, message = await queue.get()
            messages.append(message)
            temp_items.append((priority, message))
        
        # Put items back
        for item in temp_items:
            await queue.put(item)
        
        return messages
    
    def get_message_history(
        self, 
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get filtered message history"""
        filtered = self.message_history
        
        if from_agent:
            filtered = [m for m in filtered if m.from_agent == from_agent]
        if to_agent:
            filtered = [m for m in filtered if m.to_agent == to_agent]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        
        return filtered[-limit:]
    
    async def wait_for_response(
        self, 
        correlation_id: str, 
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Wait for a response message with specific correlation ID"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            for message in reversed(self.message_history):
                if message.correlation_id == correlation_id:
                    return message
            await asyncio.sleep(0.1)
        
        return None


class MessageBuilder:
    """Builder pattern for creating agent messages"""
    
    @staticmethod
    def create(
        from_agent: AgentType,
        to_agent: AgentType,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 5,
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Create a new message"""
        return AgentMessage(
            message_id=str(uuid.uuid4()),
            from_agent=from_agent.value,
            to_agent=to_agent.value if to_agent else "broadcast",
            message_type=message_type.value,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            priority=priority,
            correlation_id=correlation_id,
            metadata={}
        )
    
    @staticmethod
    def create_think_message(from_agent: AgentType, thought: str, context: Dict = None) -> AgentMessage:
        """Create a thinking message"""
        return MessageBuilder.create(
            from_agent=from_agent,
            to_agent=AgentType.ORCHESTRATOR,
            message_type=MessageType.THINK,
            content={"thought": thought, "context": context or {}},
            priority=3
        )
    
    @staticmethod
    def create_plan_message(from_agent: AgentType, plan: List[Dict], goal: str) -> AgentMessage:
        """Create a plan message"""
        return MessageBuilder.create(
            from_agent=from_agent,
            to_agent=AgentType.EXECUTION,
            message_type=MessageType.PLAN,
            content={"plan": plan, "goal": goal},
            priority=2
        )
    
    @staticmethod
    def create_execute_message(from_agent: AgentType, action: str, params: Dict) -> AgentMessage:
        """Create an execution message"""
        return MessageBuilder.create(
            from_agent=from_agent,
            to_agent=AgentType.EXECUTION,
            message_type=MessageType.EXECUTE,
            content={"action": action, "params": params},
            priority=1
        )
    
    @staticmethod
    def create_reflect_message(from_agent: AgentType, result: Any, success: bool) -> AgentMessage:
        """Create a reflection message"""
        return MessageBuilder.create(
            from_agent=from_agent,
            to_agent=AgentType.DECISION,
            message_type=MessageType.REFLECT,
            content={"result": result, "success": success},
            priority=2
        )
    
    @staticmethod
    def create_memory_store(from_agent: AgentType, key: str, data: Any, importance: int = 5) -> AgentMessage:
        """Create a memory store message"""
        return MessageBuilder.create(
            from_agent=from_agent,
            to_agent=AgentType.MEMORY,
            message_type=MessageType.MEMORY_STORE,
            content={"key": key, "data": data, "importance": importance},
            priority=4
        )
    
    @staticmethod
    def create_permission_request(from_agent: AgentType, action: str, risk_level: str, details: Dict) -> AgentMessage:
        """Create a permission request message"""
        return MessageBuilder.create(
            from_agent=from_agent,
            to_agent=AgentType.DECISION,
            message_type=MessageType.PERMISSION_REQUEST,
            content={"action": action, "risk_level": risk_level, "details": details},
            priority=1
        )


# Global message bus instance
message_bus = MessageBus()
