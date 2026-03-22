"""
JARVIS ReAct Reasoning Loop
Think → Plan → Execute → Reflect → Improve
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.core.message_bus import AgentMessage, MessageType, AgentType, MessageBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningStage(Enum):
    """Stages of the ReAct reasoning loop"""
    IDLE = "idle"
    THINK = "think"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    IMPROVE = "improve"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Thought:
    """A single thought in the reasoning process"""
    content: str
    reasoning_type: str  # "analysis", "hypothesis", "conclusion", "question"
    confidence: float = 0.5  # 0.0 - 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """A planned or executed action"""
    action_id: str
    action_type: str
    description: str
    params: Dict[str, Any]
    status: str = "pending"  # pending, executing, completed, failed
    result: Any = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Reflection:
    """Reflection on executed actions"""
    success: bool
    observations: List[str]
    lessons_learned: List[str]
    improvements: List[str]
    score: float = 0.0  # 0.0 - 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning loop iteration"""
    trace_id: str
    goal: str
    context: Dict[str, Any]
    thoughts: List[Thought] = field(default_factory=list)
    plan: List[Action] = field(default_factory=list)
    executed_actions: List[Action] = field(default_factory=list)
    reflection: Optional[Reflection] = None
    improvements: List[str] = field(default_factory=list)
    current_stage: ReasoningStage = ReasoningStage.IDLE
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_time: Optional[str] = None
    iterations: int = 0
    max_iterations: int = 5


class ReActLoop:
    """
    ReAct (Reasoning + Acting) Loop Implementation
    Implements the full reasoning cycle for autonomous decision making
    """
    
    def __init__(
        self,
        llm_client=None,
        max_iterations: int = 5,
        reflection_threshold: float = 0.7
    ):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.reflection_threshold = reflection_threshold
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self.completed_traces: List[ReasoningTrace] = []
        self.stage_handlers: Dict[ReasoningStage, Callable] = {
            ReasoningStage.THINK: self._stage_think,
            ReasoningStage.PLAN: self._stage_plan,
            ReasoningStage.EXECUTE: self._stage_execute,
            ReasoningStage.REFLECT: self._stage_reflect,
            ReasoningStage.IMPROVE: self._stage_improve,
        }
        
        # Callbacks for external integration
        self.on_stage_change: Optional[Callable] = None
        self.on_action_execute: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
    
    async def start_reasoning(
        self,
        goal: str,
        context: Dict[str, Any] = None,
        trace_id: Optional[str] = None
    ) -> ReasoningTrace:
        """Start a new reasoning loop"""
        trace = ReasoningTrace(
            trace_id=trace_id or f"trace_{datetime.utcnow().timestamp()}",
            goal=goal,
            context=context or {},
            max_iterations=self.max_iterations
        )
        
        self.active_traces[trace.trace_id] = trace
        logger.info(f"🧠 Starting ReAct loop: {goal}")
        
        # Start with THINK stage
        await self._transition_stage(trace, ReasoningStage.THINK)
        
        return trace
    
    async def continue_reasoning(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Continue an existing reasoning loop"""
        trace = self.active_traces.get(trace_id)
        if not trace:
            logger.error(f"Trace not found: {trace_id}")
            return None
        
        # Check iteration limit
        if trace.iterations >= trace.max_iterations:
            logger.warning(f"Max iterations reached for trace: {trace_id}")
            await self._transition_stage(trace, ReasoningStage.COMPLETE)
            return trace
        
        # Execute current stage
        handler = self.stage_handlers.get(trace.current_stage)
        if handler:
            await handler(trace)
        
        return trace
    
    async def _transition_stage(self, trace: ReasoningTrace, new_stage: ReasoningStage):
        """Transition to a new reasoning stage"""
        old_stage = trace.current_stage
        trace.current_stage = new_stage
        
        logger.info(f"🔄 Stage transition: {old_stage.value} → {new_stage.value}")
        
        if self.on_stage_change:
            await self.on_stage_change(trace, old_stage, new_stage)
        
        # Handle completion
        if new_stage == ReasoningStage.COMPLETE:
            trace.end_time = datetime.utcnow().isoformat()
            self.completed_traces.append(trace)
            del self.active_traces[trace.trace_id]
            
            if self.on_complete:
                await self.on_complete(trace)
        
        # Handle failure
        elif new_stage == ReasoningStage.FAILED:
            trace.end_time = datetime.utcnow().isoformat()
            del self.active_traces[trace.trace_id]
    
    async def _stage_think(self, trace: ReasoningTrace):
        """THINK stage: Analyze the problem and generate thoughts"""
        logger.info(f"🤔 THINK stage for: {trace.goal}")
        
        # Build context for thinking
        context = {
            "goal": trace.goal,
            "previous_thoughts": [t.content for t in trace.thoughts],
            "context": trace.context,
            "iteration": trace.iterations
        }
        
        # Generate thoughts using LLM
        thoughts = await self._generate_thoughts(context)
        
        for thought_data in thoughts:
            thought = Thought(
                content=thought_data.get("content", ""),
                reasoning_type=thought_data.get("type", "analysis"),
                confidence=thought_data.get("confidence", 0.5),
                metadata=thought_data.get("metadata", {})
            )
            trace.thoughts.append(thought)
            logger.info(f"💭 Thought ({thought.reasoning_type}): {thought.content[:100]}...")
        
        # Transition to PLAN stage
        await self._transition_stage(trace, ReasoningStage.PLAN)
    
    async def _stage_plan(self, trace: ReasoningTrace):
        """PLAN stage: Create action plan based on thoughts"""
        logger.info(f"📋 PLAN stage for: {trace.goal}")
        
        # Build context for planning
        context = {
            "goal": trace.goal,
            "thoughts": [
                {"content": t.content, "type": t.reasoning_type, "confidence": t.confidence}
                for t in trace.thoughts
            ],
            "available_actions": trace.context.get("available_actions", []),
            "constraints": trace.context.get("constraints", [])
        }
        
        # Generate plan using LLM
        plan_actions = await self._generate_plan(context)
        
        trace.plan = []
        for i, action_data in enumerate(plan_actions):
            action = Action(
                action_id=f"action_{trace.trace_id}_{i}",
                action_type=action_data.get("type", "unknown"),
                description=action_data.get("description", ""),
                params=action_data.get("params", {})
            )
            trace.plan.append(action)
            logger.info(f"📌 Planned action: {action.description}")
        
        # Transition to EXECUTE stage
        await self._transition_stage(trace, ReasoningStage.EXECUTE)
    
    async def _stage_execute(self, trace: ReasoningTrace):
        """EXECUTE stage: Execute planned actions"""
        logger.info(f"⚡ EXECUTE stage for: {trace.goal}")
        
        for action in trace.plan:
            if action.status != "pending":
                continue
            
            action.status = "executing"
            logger.info(f"🔧 Executing: {action.description}")
            
            start_time = datetime.utcnow()
            
            try:
                # Execute the action
                result = await self._execute_action(action, trace)
                
                action.result = result
                action.status = "completed"
                action.execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                trace.executed_actions.append(action)
                
                if self.on_action_execute:
                    await self.on_action_execute(trace, action, result)
                
                logger.info(f"✅ Action completed: {action.description}")
                
            except Exception as e:
                action.status = "failed"
                action.result = {"error": str(e)}
                action.execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(f"❌ Action failed: {action.description} - {e}")
        
        # Transition to REFLECT stage
        await self._transition_stage(trace, ReasoningStage.REFLECT)
    
    async def _stage_reflect(self, trace: ReasoningTrace):
        """REFLECT stage: Evaluate execution results"""
        logger.info(f"🪞 REFLECT stage for: {trace.goal}")
        
        # Build context for reflection
        context = {
            "goal": trace.goal,
            "thoughts": [t.content for t in trace.thoughts],
            "actions": [
                {
                    "description": a.description,
                    "status": a.status,
                    "result": a.result,
                    "execution_time": a.execution_time
                }
                for a in trace.executed_actions
            ]
        }
        
        # Generate reflection using LLM
        reflection_data = await self._generate_reflection(context)
        
        reflection = Reflection(
            success=reflection_data.get("success", False),
            observations=reflection_data.get("observations", []),
            lessons_learned=reflection_data.get("lessons_learned", []),
            improvements=reflection_data.get("improvements", []),
            score=reflection_data.get("score", 0.0)
        )
        
        trace.reflection = reflection
        trace.iterations += 1
        
        logger.info(f"🪞 Reflection: success={reflection.success}, score={reflection.score}")
        
        for obs in reflection.observations:
            logger.info(f"  👁️ Observation: {obs}")
        
        # Decide next stage
        if reflection.score >= self.reflection_threshold:
            await self._transition_stage(trace, ReasoningStage.COMPLETE)
        elif trace.iterations >= trace.max_iterations:
            await self._transition_stage(trace, ReasoningStage.COMPLETE)
        else:
            await self._transition_stage(trace, ReasoningStage.IMPROVE)
    
    async def _stage_improve(self, trace: ReasoningTrace):
        """IMPROVE stage: Apply lessons learned and iterate"""
        logger.info(f"📈 IMPROVE stage for: {trace.goal}")
        
        if trace.reflection:
            trace.improvements.extend(trace.reflection.improvements)
            
            # Update context with lessons learned
            trace.context["lessons_learned"] = trace.reflection.lessons_learned
            trace.context["previous_attempts"] = trace.iterations
        
        logger.info(f"📈 Improvements: {len(trace.improvements)}")
        
        # Start next iteration
        await self._transition_stage(trace, ReasoningStage.THINK)
    
    # LLM Integration methods (to be implemented with actual LLM)
    async def _generate_thoughts(self, context: Dict) -> List[Dict]:
        """Generate thoughts using LLM"""
        if self.llm_client:
            return await self.llm_client.generate_thoughts(context)
        
        # Default implementation
        return [
            {
                "content": f"Analyzing goal: {context['goal']}",
                "type": "analysis",
                "confidence": 0.8
            },
            {
                "content": "Breaking down the problem into manageable steps",
                "type": "hypothesis",
                "confidence": 0.7
            }
        ]
    
    async def _generate_plan(self, context: Dict) -> List[Dict]:
        """Generate action plan using LLM"""
        if self.llm_client:
            return await self.llm_client.generate_plan(context)
        
        # Default implementation
        return [
            {
                "type": "research",
                "description": "Gather information about the task",
                "params": {"query": context["goal"]}
            },
            {
                "type": "execute",
                "description": "Execute the main task",
                "params": {"task": context["goal"]}
            }
        ]
    
    async def _execute_action(self, action: Action, trace: ReasoningTrace) -> Any:
        """Execute a single action"""
        # This should be integrated with the Execution Agent
        # For now, return a placeholder
        return {
            "action_type": action.action_type,
            "params": action.params,
            "status": "simulated"
        }
    
    async def _generate_reflection(self, context: Dict) -> Dict:
        """Generate reflection using LLM"""
        if self.llm_client:
            return await self.llm_client.generate_reflection(context)
        
        # Default implementation
        return {
            "success": True,
            "observations": ["Task completed successfully"],
            "lessons_learned": ["Plan was effective"],
            "improvements": [],
            "score": 0.8
        }
    
    # Utility methods
    def get_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Get a reasoning trace by ID"""
        return self.active_traces.get(trace_id) or next(
            (t for t in self.completed_traces if t.trace_id == trace_id),
            None
        )
    
    def get_all_traces(self) -> List[ReasoningTrace]:
        """Get all traces (active and completed)"""
        return list(self.active_traces.values()) + self.completed_traces
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get a summary of a reasoning trace"""
        trace = self.get_trace(trace_id)
        if not trace:
            return {}
        
        return {
            "trace_id": trace.trace_id,
            "goal": trace.goal,
            "current_stage": trace.current_stage.value,
            "iterations": trace.iterations,
            "thoughts_count": len(trace.thoughts),
            "actions_planned": len(trace.plan),
            "actions_executed": len(trace.executed_actions),
            "reflection_score": trace.reflection.score if trace.reflection else None,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "duration": self._calculate_duration(trace)
        }
    
    def _calculate_duration(self, trace: ReasoningTrace) -> float:
        """Calculate trace duration in seconds"""
        start = datetime.fromisoformat(trace.start_time)
        end = datetime.fromisoformat(trace.end_time) if trace.end_time else datetime.utcnow()
        return (end - start).total_seconds()


class ReasoningMessageAdapter:
    """
    Adapter to integrate ReAct loop with the message bus
    Converts between AgentMessages and ReAct loop operations
    """
    
    def __init__(self, react_loop: ReActLoop):
        self.react_loop = react_loop
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming agent messages"""
        msg_type = message.message_type
        
        if msg_type == MessageType.THINK.value:
            # Start or continue thinking
            trace_id = message.content.get("trace_id")
            if trace_id:
                trace = await self.react_loop.continue_reasoning(trace_id)
            else:
                trace = await self.react_loop.start_reasoning(
                    goal=message.content.get("goal", ""),
                    context=message.content.get("context", {})
                )
            
            return self._create_response(message, trace)
        
        elif msg_type == MessageType.PLAN.value:
            # Transition to planning
            trace_id = message.content.get("trace_id")
            trace = self.react_loop.get_trace(trace_id)
            if trace:
                await self.react_loop._transition_stage(trace, ReasoningStage.PLAN)
                trace = await self.react_loop.continue_reasoning(trace_id)
            return self._create_response(message, trace)
        
        elif msg_type == MessageType.EXECUTE.value:
            # Execute actions
            trace_id = message.content.get("trace_id")
            trace = self.react_loop.get_trace(trace_id)
            if trace:
                await self.react_loop._transition_stage(trace, ReasoningStage.EXECUTE)
                trace = await self.react_loop.continue_reasoning(trace_id)
            return self._create_response(message, trace)
        
        elif msg_type == MessageType.REFLECT.value:
            # Reflect on results
            trace_id = message.content.get("trace_id")
            trace = self.react_loop.get_trace(trace_id)
            if trace:
                await self.react_loop._transition_stage(trace, ReasoningStage.REFLECT)
                trace = await self.react_loop.continue_reasoning(trace_id)
            return self._create_response(message, trace)
        
        return None
    
    def _create_response(
        self,
        original_message: AgentMessage,
        trace: ReasoningTrace
    ) -> AgentMessage:
        """Create a response message"""
        return MessageBuilder.create(
            from_agent=AgentType.ORCHESTRATOR,
            to_agent=AgentType(original_message.from_agent),
            message_type=MessageType.RESPONSE,
            content={
                "success": True,
                "trace": self.react_loop.get_trace_summary(trace.trace_id),
                "stage": trace.current_stage.value
            },
            priority=2,
            correlation_id=original_message.message_id
        )
