"""
JARVIS Orchestrator - Central Controller for the 6-Agent System
Manages agent lifecycle, coordinates workflows, handles user requests
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

from app.core.message_bus import (
    MessageBus, AgentMessage, MessageType, AgentType, MessageBuilder, message_bus
)
from app.core.base_agent import BaseAgent
from app.reasoning.graph_engine import LangGraphEngine

# Import all agents
from app.agents.planner.planner_agent import PlannerAgent
from app.agents.decision.decision_agent import DecisionAgent
from app.agents.execution.execution_agent import ExecutionAgent
from app.agents.memory.memory_agent import MemoryAgent
from app.agents.communication.communication_agent import CommunicationAgent
from app.agents.automation.automation_agent import AutomationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class Orchestrator:
    """
    JARVIS Orchestrator - The brain of the AI operating system
    
    Responsibilities:
    - Initialize and manage all 6 agents
    - Route messages between agents
    - Coordinate the ReAct reasoning loop
    - Handle user requests
    - Manage autonomous mode
    - Monitor system health
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.message_bus = message_bus
        
        # System state
        self.state = SystemState.INITIALIZING
        self.started_at = None
        
        # Agents
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.agent_configs = config.get("agents", {}) if config else {}
        
        # LangGraph Engine
        self.graph_engine = LangGraphEngine(self)
        self.active_reasoning_traces: Dict[str, Any] = {}
        
        # User session
        self.user_id: Optional[str] = None
        self.session_context: Dict[str, Any] = {}
        
        # Autonomous mode
        self.autonomous_mode = False
        self.autonomous_tasks: asyncio.Queue = asyncio.Queue()
        self._autonomous_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_user_response: Optional[Callable] = None
        
        # Metrics
        self.metrics = {
            "requests_processed": 0,
            "agents_initialized": 0,
            "errors": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the entire JARVIS system"""
        logger.info("🚀 Initializing JARVIS System...")
        self._set_state(SystemState.INITIALIZING)
        
        try:
            # Start message bus
            await self.message_bus.start()
            
            # Initialize all agents
            await self._initialize_agents()
            
            # Start all agents
            await self._start_agents()
            
            # Graph is state machine and statically compiled
            # React components removed
            self.started_at = datetime.utcnow().isoformat()
            self._set_state(SystemState.READY)
            
            logger.info("✅ JARVIS System initialized successfully")
            logger.info(f"   Agents: {list(self.agents.keys())}")
            
            return True
            
        except Exception as e:
            self._set_state(SystemState.ERROR)
            logger.error(f"❌ JARVIS initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_agents(self):
        """Initialize all 6 agents"""
        logger.info("🤖 Initializing agents...")
        
        agent_classes = {
            AgentType.PLANNER: PlannerAgent,
            AgentType.DECISION: DecisionAgent,
            AgentType.EXECUTION: ExecutionAgent,
            AgentType.MEMORY: MemoryAgent,
            AgentType.COMMUNICATION: CommunicationAgent,
            AgentType.AUTOMATION: AutomationAgent
        }
        
        for agent_type, agent_class in agent_classes.items():
            try:
                config = self.agent_configs.get(agent_type.value, {})
                agent = agent_class(config=config)
                
                success = await agent.initialize()
                if success:
                    self.agents[agent_type] = agent
                    self.metrics["agents_initialized"] += 1
                    logger.info(f"  ✅ {agent_type.value} agent ready")
                else:
                    logger.error(f"  ❌ {agent_type.value} agent failed to initialize")
                    
            except Exception as e:
                logger.error(f"  ❌ {agent_type.value} agent error: {e}")
    
    async def _start_agents(self):
        """Start all initialized agents"""
        logger.info("▶️ Starting agents...")
        
        for agent_type, agent in self.agents.items():
            try:
                await agent.start()
            except Exception as e:
                logger.error(f"  ❌ Failed to start {agent_type.value}: {e}")
    
    def _setup_react_integration(self):
        """Set up ReAct loop integration with message bus"""
        self.react_loop.on_stage_change = self._on_reasoning_stage_change
        self.react_loop.on_action_execute = self._on_action_execute
        self.react_loop.on_complete = self._on_reasoning_complete
    
    async def shutdown(self):
        """Shutdown the entire system"""
        logger.info("🛑 Shutting down JARVIS System...")
        self._set_state(SystemState.SHUTDOWN)
        
        # Stop autonomous mode
        await self.stop_autonomous_mode()
        
        # Stop all agents
        for agent_type, agent in self.agents.items():
            try:
                await agent.stop()
                logger.info(f"  ⏹️ {agent_type.value} agent stopped")
            except Exception as e:
                logger.error(f"  ❌ Error stopping {agent_type.value}: {e}")
        
        # Stop message bus
        await self.message_bus.stop()
        
        logger.info("✅ JARVIS System shutdown complete")
    
    # User request handling
    async def process_user_request(
        self,
        request: str,
        context: Dict[str, Any] = None,
        use_voice: bool = False
    ) -> Dict[str, Any]:
        """Process a user request through the full agent pipeline"""
        logger.info(f"👤 User request: {request}")
        self._set_state(SystemState.PROCESSING)
        self.metrics["requests_processed"] += 1
        
        context = context or {}
        
        try:
            # Step 1: Store user input in memory
            memory_agent = self.agents.get(AgentType.MEMORY)
            if memory_agent:
                memory_agent.add_to_conversation_history("user", request, context)
                await memory_agent.store(
                    content={"request": request, "context": context},
                    memory_type="context",
                    importance=5,
                    tags=["user_request"]
                )
            
            # Step 2: Retrieve relevant context
            relevant_memories = []
            if memory_agent:
                relevant_memories = await memory_agent.retrieve(
                    query=request,
                    limit=5
                )
            
            # Step 3, 4, 5: Run LangGraph Agent explicitly
            result_state = await self.graph_engine.run(
                goal=request,
                context={
                    "user_request": True,
                    "relevant_memories": [m.to_dict() for m in relevant_memories],
                    "conversation_history": memory_agent.get_conversation_context() if memory_agent else [],
                    "user_profile": memory_agent.get_user_profile() if memory_agent else {}
                }
            )
            
            execution_result = result_state.get("executed_actions", [])
            
            # Step 6: Store result in memory
            if memory_agent:
                await memory_agent.store(
                    content={
                        "request": request,
                        "result": execution_result
                    },
                    memory_type="experience",
                    importance=6,
                    tags=["completed_request"]
                )
            
            # Step 7: Communicate result to user
            await self._communicate_result({"results": execution_result}, use_voice)
            
            self._set_state(SystemState.READY)
            
            return {
                "success": True,
                "trace_id": "langgraph_" + str(datetime.utcnow().timestamp()),
                "result": execution_result,
                "thoughts": result_state.get("thoughts", []),
                "actions_executed": len(execution_result)
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"❌ Request processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            self._set_state(SystemState.ERROR)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_reasoning_result(self, trace) -> Dict[str, Any]:
        """Execute the result of reasoning"""
        results = []
        
        for action in trace.executed_actions:
            if action.action_type == "delegate_to_agent":
                agent_type = AgentType(action.params.get("agent"))
                task = action.params.get("task", {})
                
                agent = self.agents.get(agent_type)
                if agent:
                    # Send task to agent
                    message = await agent.send_message(
                        to_agent=agent_type,
                        message_type=MessageType.EXECUTE,
                        content=task,
                        priority=2
                    )
                    
                    # Wait for response
                    response = await self.message_bus.wait_for_response(
                        correlation_id=message.message_id,
                        timeout=60.0
                    )
                    
                    results.append({
                        "agent": agent_type.value,
                        "result": response.content if response else None
                    })
        
        return {
            "executed_actions": len(trace.executed_actions),
            "results": results
        }
    
    async def _communicate_result(self, result: Dict, use_voice: bool = False):
        """Communicate result to user"""
        comm_agent = self.agents.get(AgentType.COMMUNICATION)
        if not comm_agent:
            return
        
        # Format result for user
        message = self._format_result_for_user(result)
        
        if use_voice:
            await comm_agent.speak(message)
        else:
            await comm_agent.send_text(message)
    
    def _format_result_for_user(self, result: Dict) -> str:
        """Format execution result for user communication"""
        parts = []
        
        for r in result.get("results", []):
            agent_result = r.get("result", {})
            if agent_result and agent_result.get("success"):
                parts.append(f"✅ {r['agent']}: Done")
            else:
                parts.append(f"❌ {r['agent']}: Failed")
        
        return "\n".join(parts) if parts else "Task completed."
    
    # Autonomous mode
    async def start_autonomous_mode(self):
        """Start autonomous operation mode"""
        if self.autonomous_mode:
            return
        
        logger.info("🤖 Starting autonomous mode...")
        self.autonomous_mode = True
        self._set_state(SystemState.AUTONOMOUS)
        
        self._autonomous_task = asyncio.create_task(self._autonomous_loop())
        
        # Notify user
        comm_agent = self.agents.get(AgentType.COMMUNICATION)
        if comm_agent:
            await comm_agent.send_text(
                "🤖 Autonomous mode activated. I'll work on your tasks in the background.",
                priority="high"
            )
    
    async def stop_autonomous_mode(self):
        """Stop autonomous operation mode"""
        if not self.autonomous_mode:
            return
        
        logger.info("🛑 Stopping autonomous mode...")
        self.autonomous_mode = False
        
        if self._autonomous_task:
            self._autonomous_task.cancel()
            try:
                await self._autonomous_task
            except asyncio.CancelledError:
                pass
        
        self._set_state(SystemState.READY)
        
        # Notify user
        comm_agent = self.agents.get(AgentType.COMMUNICATION)
        if comm_agent:
            await comm_agent.send_text(
                "🛑 Autonomous mode deactivated.",
                priority="normal"
            )
    
    async def _autonomous_loop(self):
        """Main autonomous operation loop"""
        while self.autonomous_mode:
            try:
                # Run money automation periodically
                automation_agent = self.agents.get(AgentType.AUTOMATION)
                if automation_agent:
                    result = await automation_agent.run_money_automation()
                    
                    # Notify if applications were sent
                    if result.get("applications_sent", 0) > 0:
                        comm_agent = self.agents.get(AgentType.COMMUNICATION)
                        if comm_agent:
                            await comm_agent.send_text(
                                f"💰 Applied to {result['applications_sent']} jobs today!",
                                priority="normal"
                            )
                
                # Wait before next iteration
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"❌ Autonomous loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    # ReAct loop callbacks
    async def _on_reasoning_stage_change(self, trace, old_stage, new_stage):
        """Handle reasoning stage changes"""
        logger.debug(f"Reasoning: {old_stage.value} → {new_stage.value}")
    
    async def _on_action_execute(self, trace, action, result):
        """Handle action execution during reasoning"""
        logger.debug(f"Executed: {action.description}")
    
    async def _on_reasoning_complete(self, trace):
        """Handle reasoning completion"""
        logger.info(f"✅ Reasoning complete: {trace.goal}")
        
        if trace.trace_id in self.active_reasoning_traces:
            del self.active_reasoning_traces[trace.trace_id]
    
    # Special commands
    async def handle_special_command(self, command: str, params: Dict = None) -> Dict:
        """Handle special system commands"""
        params = params or {}
        
        if command == "make_me_money":
            return await self._cmd_make_me_money(params)
        elif command == "start_voice_session":
            return await self._cmd_start_voice(params)
        elif command == "get_status":
            return await self._cmd_get_status(params)
        elif command == "toggle_autonomous":
            return await self._cmd_toggle_autonomous(params)
        elif command == "get_memory":
            return await self._cmd_get_memory(params)
        elif command == "clear_memory":
            return await self._cmd_clear_memory(params)
        else:
            return {"error": f"Unknown command: {command}"}
    
    async def _cmd_make_me_money(self, params: Dict) -> Dict:
        """Execute 'Make me money today' command"""
        automation_agent = self.agents.get(AgentType.AUTOMATION)
        if not automation_agent:
            return {"error": "Automation agent not available"}
        
        result = await automation_agent.run_money_automation()
        
        return {
            "command": "make_me_money",
            "result": result
        }
    
    async def _cmd_start_voice(self, params: Dict) -> Dict:
        """Start voice session"""
        comm_agent = self.agents.get(AgentType.COMMUNICATION)
        if not comm_agent:
            return {"error": "Communication agent not available"}
        
        language = params.get("language", "en")
        session = await comm_agent.start_voice_session(language)
        
        return {
            "command": "start_voice_session",
            "session_id": session.session_id,
            "status": "started"
        }
    
    async def _cmd_get_status(self, params: Dict) -> Dict:
        """Get system status"""
        return {
            "command": "get_status",
            "system_state": self.state.value,
            "agents": {
                agent_type.value: agent.get_metrics()
                for agent_type, agent in self.agents.items()
            },
            "metrics": self.metrics,
            "autonomous_mode": self.autonomous_mode
        }
    
    async def _cmd_toggle_autonomous(self, params: Dict) -> Dict:
        """Toggle autonomous mode"""
        if self.autonomous_mode:
            await self.stop_autonomous_mode()
            return {"command": "toggle_autonomous", "mode": "off"}
        else:
            await self.start_autonomous_mode()
            return {"command": "toggle_autonomous", "mode": "on"}
    
    async def _cmd_get_memory(self, params: Dict) -> Dict:
        """Get memory information"""
        memory_agent = self.agents.get(AgentType.MEMORY)
        if not memory_agent:
            return {"error": "Memory agent not available"}
        
        return {
            "command": "get_memory",
            "stats": memory_agent.get_memory_stats(),
            "user_profile": memory_agent.get_user_profile()
        }
    
    async def _cmd_clear_memory(self, params: Dict) -> Dict:
        """Clear memory"""
        memory_agent = self.agents.get(AgentType.MEMORY)
        if not memory_agent:
            return {"error": "Memory agent not available"}
        
        # Implementation depends on memory agent
        return {
            "command": "clear_memory",
            "status": "not_implemented"
        }
    
    # State management
    def _set_state(self, new_state: SystemState):
        """Update system state"""
        old_state = self.state
        self.state = new_state
        
        if self.on_state_change:
            asyncio.create_task(self.on_state_change(old_state, new_state))
        
        logger.info(f"🔄 System state: {old_state.value} → {new_state.value}")
    
    def get_state(self) -> SystemState:
        """Get current system state"""
        return self.state
    
    def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Get an agent by type"""
        return self.agents.get(agent_type)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "state": self.state.value,
            "started_at": self.started_at,
            "agents": {
                agent_type.value: {
                    "state": agent.get_state(),
                    "metrics": agent.get_metrics()
                }
                for agent_type, agent in self.agents.items()
            },
            "metrics": self.metrics,
            "autonomous_mode": self.autonomous_mode,
            "active_traces": len(self.active_reasoning_traces)
        }
