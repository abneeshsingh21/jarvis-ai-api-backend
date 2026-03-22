"""
JARVIS Planner Agent
Breaks down complex goals into actionable steps
Creates strategic plans for task execution
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.base_agent import BaseAgent, AgentState
from app.core.message_bus import (
    AgentMessage, MessageType, AgentType, MessageBuilder, message_bus
)
from app.reasoning.react_loop import ReActLoop, ReasoningStage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskBreakdown:
    """Represents a broken-down task"""
    def __init__(
        self,
        task_id: str,
        description: str,
        priority: int = 5,
        dependencies: List[str] = None,
        estimated_duration: int = 0,
        required_agents: List[str] = None,
        success_criteria: List[str] = None
    ):
        self.task_id = task_id
        self.description = description
        self.priority = priority
        self.dependencies = dependencies or []
        self.estimated_duration = estimated_duration
        self.required_agents = required_agents or []
        self.success_criteria = success_criteria or []
        self.status = "pending"
        self.created_at = datetime.utcnow().isoformat()


class StrategicPlan:
    """A complete strategic plan"""
    def __init__(
        self,
        plan_id: str,
        goal: str,
        context: Dict[str, Any] = None
    ):
        self.plan_id = plan_id
        self.goal = goal
        self.context = context or {}
        self.tasks: List[TaskBreakdown] = []
        self.timeline: Dict[str, Any] = {}
        self.risks: List[Dict] = []
        self.contingencies: List[Dict] = []
        self.created_at = datetime.utcnow().isoformat()
        self.status = "draft"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "context": self.context,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "priority": t.priority,
                    "dependencies": t.dependencies,
                    "estimated_duration": t.estimated_duration,
                    "required_agents": t.required_agents,
                    "success_criteria": t.success_criteria,
                    "status": t.status
                }
                for t in self.tasks
            ],
            "timeline": self.timeline,
            "risks": self.risks,
            "contingencies": self.contingencies,
            "created_at": self.created_at,
            "status": self.status
        }


class PlannerAgent(BaseAgent):
    """
    Planner Agent: Breaks down goals into actionable plans
    
    Capabilities:
    - Goal decomposition
    - Task prioritization
    - Dependency mapping
    - Timeline estimation
    - Risk assessment
    - Resource allocation
    """
    
    def __init__(self, llm_client=None, config: Dict = None):
        super().__init__(AgentType.PLANNER, config=config)
        self.llm_client = llm_client
        self.active_plans: Dict[str, StrategicPlan] = {}
        self.plan_history: List[StrategicPlan] = []
        self.planning_strategies = {
            "sequential": self._plan_sequential,
            "parallel": self._plan_parallel,
            "adaptive": self._plan_adaptive,
            "milestone": self._plan_milestone
        }
        self.react_loop = ReActLoop(llm_client=llm_client)
    
    async def _initialize(self) -> bool:
        """Initialize the Planner Agent"""
        logger.info("📋 Planner Agent initializing...")
        
        # Register message handlers
        self.register_handler(MessageType.REQUEST, self._handle_planning_request)
        self.register_handler(MessageType.PLAN, self._handle_plan_message)
        
        # Set up ReAct loop callbacks
        self.react_loop.on_stage_change = self._on_reasoning_stage_change
        self.react_loop.on_complete = self._on_reasoning_complete
        
        logger.info("✅ Planner Agent initialized")
        return True
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.active_plans.clear()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle generic messages"""
        logger.info(f"📥 Planner received: {message.message_type}")
    
    async def _handle_planning_request(self, message: AgentMessage):
        """Handle planning requests"""
        content = message.content
        goal = content.get("goal", "")
        strategy = content.get("strategy", "adaptive")
        context = content.get("context", {})
        
        logger.info(f"🎯 Planning request: {goal}")
        
        # Create a plan
        plan = await self.create_plan(goal, strategy, context)
        
        # Store the plan
        self.active_plans[plan.plan_id] = plan
        
        # Send response
        await self.send_response(
            original_message=message,
            content={
                "plan_id": plan.plan_id,
                "plan": plan.to_dict(),
                "task_count": len(plan.tasks)
            }
        )
        
        # Broadcast plan created
        await self.broadcast(
            message_type=MessageType.STATUS_UPDATE,
            content={
                "event": "plan_created",
                "plan_id": plan.plan_id,
                "goal": goal,
                "tasks": len(plan.tasks)
            }
        )
    
    async def _handle_plan_message(self, message: AgentMessage):
        """Handle plan messages from other agents"""
        content = message.content
        plan_data = content.get("plan", [])
        goal = content.get("goal", "")
        
        logger.info(f"📋 Received plan for: {goal}")
        
        # Convert to StrategicPlan
        plan_id = f"plan_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal)
        
        for i, task_data in enumerate(plan_data):
            task = TaskBreakdown(
                task_id=task_data.get("id", f"task_{i}"),
                description=task_data.get("description", ""),
                priority=task_data.get("priority", 5),
                dependencies=task_data.get("dependencies", []),
                estimated_duration=task_data.get("duration", 0),
                required_agents=task_data.get("agents", []),
                success_criteria=task_data.get("criteria", [])
            )
            plan.tasks.append(task)
        
        self.active_plans[plan_id] = plan
    
    async def create_plan(
        self,
        goal: str,
        strategy: str = "adaptive",
        context: Dict[str, Any] = None
    ) -> StrategicPlan:
        """Create a strategic plan for a goal"""
        logger.info(f"📋 Creating plan for: {goal}")
        
        plan_id = f"plan_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal, context)
        
        # Use ReAct loop for complex planning
        if context and context.get("use_reasoning", True):
            trace = await self.react_loop.start_reasoning(
                goal=f"Create a detailed plan for: {goal}",
                context={
                    "strategy": strategy,
                    "user_context": context,
                    "available_agents": [a.value for a in AgentType]
                }
            )
            
            # Wait for planning to complete
            while trace.current_stage != ReasoningStage.COMPLETE:
                await self.react_loop.continue_reasoning(trace.trace_id)
                await asyncio.sleep(0.1)
            
            # Extract plan from reasoning trace
            plan = self._extract_plan_from_trace(trace, goal)
        else:
            # Use direct planning strategy
            planner = self.planning_strategies.get(strategy, self._plan_adaptive)
            plan = await planner(goal, context)
        
        # Assess risks
        plan.risks = await self._assess_risks(plan)
        
        # Create contingencies
        plan.contingencies = await self._create_contingencies(plan)
        
        # Build timeline
        plan.timeline = self._build_timeline(plan)
        
        plan.status = "ready"
        logger.info(f"✅ Plan created: {len(plan.tasks)} tasks")
        
        return plan
    
    async def _plan_sequential(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> StrategicPlan:
        """Create a sequential plan (tasks in order)"""
        plan_id = f"plan_seq_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal, context)
        
        # Use LLM to break down goal
        tasks_data = await self._llm_breakdown(goal, context)
        
        prev_task_id = None
        for i, task_data in enumerate(tasks_data):
            task = TaskBreakdown(
                task_id=f"task_{i}",
                description=task_data["description"],
                priority=task_data.get("priority", 5),
                dependencies=[prev_task_id] if prev_task_id else [],
                estimated_duration=task_data.get("duration", 30),
                required_agents=task_data.get("agents", ["execution"]),
                success_criteria=task_data.get("criteria", [])
            )
            plan.tasks.append(task)
            prev_task_id = task.task_id
        
        return plan
    
    async def _plan_parallel(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> StrategicPlan:
        """Create a parallel plan (independent tasks)"""
        plan_id = f"plan_par_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal, context)
        
        tasks_data = await self._llm_breakdown(goal, context)
        
        for i, task_data in enumerate(tasks_data):
            task = TaskBreakdown(
                task_id=f"task_{i}",
                description=task_data["description"],
                priority=task_data.get("priority", 5),
                dependencies=[],  # No dependencies for parallel execution
                estimated_duration=task_data.get("duration", 30),
                required_agents=task_data.get("agents", ["execution"]),
                success_criteria=task_data.get("criteria", [])
            )
            plan.tasks.append(task)
        
        return plan
    
    async def _plan_adaptive(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> StrategicPlan:
        """Create an adaptive plan (mix of sequential and parallel)"""
        plan_id = f"plan_adapt_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal, context)
        
        tasks_data = await self._llm_breakdown(goal, context)
        
        # Group tasks by phase
        phases = {}
        for task_data in tasks_data:
            phase = task_data.get("phase", "main")
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(task_data)
        
        # Create tasks with phase dependencies
        prev_phase_last_task = None
        task_counter = 0
        
        for phase_name, phase_tasks in phases.items():
            phase_first_task = None
            
            for task_data in phase_tasks:
                dependencies = []
                if prev_phase_last_task:
                    dependencies.append(prev_phase_last_task)
                
                task = TaskBreakdown(
                    task_id=f"task_{task_counter}",
                    description=task_data["description"],
                    priority=task_data.get("priority", 5),
                    dependencies=dependencies,
                    estimated_duration=task_data.get("duration", 30),
                    required_agents=task_data.get("agents", ["execution"]),
                    success_criteria=task_data.get("criteria", [])
                )
                plan.tasks.append(task)
                
                if phase_first_task is None:
                    phase_first_task = task.task_id
                
                task_counter += 1
            
            if plan.tasks:
                prev_phase_last_task = plan.tasks[-1].task_id
        
        return plan
    
    async def _plan_milestone(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> StrategicPlan:
        """Create a milestone-based plan"""
        plan_id = f"plan_mil_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal, context)
        
        # Define milestones
        milestones = [
            {"name": "research", "description": "Research and gather information"},
            {"name": "prepare", "description": "Prepare resources and setup"},
            {"name": "execute", "description": "Execute main task"},
            {"name": "validate", "description": "Validate results"},
            {"name": "deliver", "description": "Deliver final output"}
        ]
        
        prev_milestone_task = None
        
        for i, milestone in enumerate(milestones):
            task = TaskBreakdown(
                task_id=f"milestone_{i}",
                description=f"{milestone['name']}: {milestone['description']}",
                priority=5,
                dependencies=[prev_milestone_task] if prev_milestone_task else [],
                estimated_duration=30,
                required_agents=["execution"],
                success_criteria=[f"{milestone['name']} completed successfully"]
            )
            plan.tasks.append(task)
            prev_milestone_task = task.task_id
        
        return plan
    
    async def _llm_breakdown(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> List[Dict]:
        """Use LLM to break down goal into tasks"""
        if self.llm_client:
            return await self.llm_client.breakdown_goal(goal, context)
        
        # Default breakdown
        return [
            {
                "description": f"Analyze requirements for: {goal}",
                "priority": 1,
                "duration": 15,
                "agents": ["execution"],
                "criteria": ["Requirements documented"]
            },
            {
                "description": f"Execute main task: {goal}",
                "priority": 2,
                "duration": 60,
                "agents": ["execution", "automation"],
                "criteria": ["Task completed"]
            },
            {
                "description": f"Validate and deliver results for: {goal}",
                "priority": 3,
                "duration": 15,
                "agents": ["execution"],
                "criteria": ["Results validated"]
            }
        ]
    
    async def _assess_risks(self, plan: StrategicPlan) -> List[Dict]:
        """Assess risks for the plan"""
        risks = []
        
        # Check for long tasks
        long_tasks = [t for t in plan.tasks if t.estimated_duration > 60]
        if long_tasks:
            risks.append({
                "type": "duration",
                "description": f"{len(long_tasks)} tasks exceed 60 minutes",
                "severity": "medium",
                "mitigation": "Consider breaking down long tasks"
            })
        
        # Check for many dependencies
        high_dependency_tasks = [t for t in plan.tasks if len(t.dependencies) > 2]
        if high_dependency_tasks:
            risks.append({
                "type": "dependency",
                "description": f"{len(high_dependency_tasks)} tasks have many dependencies",
                "severity": "medium",
                "mitigation": "Review and simplify dependencies"
            })
        
        # Check for external dependencies
        if plan.context.get("external_dependencies"):
            risks.append({
                "type": "external",
                "description": "Plan relies on external dependencies",
                "severity": "high",
                "mitigation": "Have backup plans for external failures"
            })
        
        return risks
    
    async def _create_contingencies(self, plan: StrategicPlan) -> List[Dict]:
        """Create contingency plans"""
        contingencies = []
        
        for risk in plan.risks:
            contingencies.append({
                "trigger": risk["description"],
                "action": risk["mitigation"],
                "fallback": "Escalate to user for decision"
            })
        
        return contingencies
    
    def _build_timeline(self, plan: StrategicPlan) -> Dict[str, Any]:
        """Build execution timeline"""
        total_duration = sum(t.estimated_duration for t in plan.tasks)
        
        # Calculate critical path (simplified)
        critical_path = []
        current_time = 0
        
        for task in sorted(plan.tasks, key=lambda t: t.priority):
            critical_path.append({
                "task_id": task.task_id,
                "start": current_time,
                "end": current_time + task.estimated_duration
            })
            current_time += task.estimated_duration
        
        return {
            "total_duration": total_duration,
            "critical_path": critical_path,
            "parallel_opportunities": len([t for t in plan.tasks if not t.dependencies])
        }
    
    def _extract_plan_from_trace(
        self,
        trace,
        goal: str
    ) -> StrategicPlan:
        """Extract a plan from a completed reasoning trace"""
        plan_id = f"plan_react_{datetime.utcnow().timestamp()}"
        plan = StrategicPlan(plan_id, goal)
        
        # Extract tasks from executed actions
        for action in trace.executed_actions:
            if action.action_type == "plan_task":
                task = TaskBreakdown(
                    task_id=action.params.get("task_id", f"task_{len(plan.tasks)}"),
                    description=action.params.get("description", ""),
                    priority=action.params.get("priority", 5),
                    dependencies=action.params.get("dependencies", []),
                    estimated_duration=action.params.get("duration", 30),
                    required_agents=action.params.get("agents", ["execution"])
                )
                plan.tasks.append(task)
        
        return plan
    
    async def _on_reasoning_stage_change(self, trace, old_stage, new_stage):
        """Callback for reasoning stage changes"""
        logger.debug(f"Reasoning stage: {old_stage.value} → {new_stage.value}")
    
    async def _on_reasoning_complete(self, trace):
        """Callback for reasoning completion"""
        logger.info(f"✅ Reasoning complete for: {trace.goal}")
    
    def get_plan(self, plan_id: str) -> Optional[StrategicPlan]:
        """Get a plan by ID"""
        return self.active_plans.get(plan_id)
    
    def get_next_task(self, plan_id: str) -> Optional[TaskBreakdown]:
        """Get the next executable task from a plan"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return None
        
        # Find tasks with no pending dependencies
        for task in plan.tasks:
            if task.status == "pending":
                # Check if all dependencies are complete
                deps_complete = all(
                    self._get_task_status(plan_id, dep) == "completed"
                    for dep in task.dependencies
                )
                if deps_complete:
                    return task
        
        return None
    
    def _get_task_status(self, plan_id: str, task_id: str) -> str:
        """Get the status of a task"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return "unknown"
        
        for task in plan.tasks:
            if task.task_id == task_id:
                return task.status
        
        return "unknown"
    
    def update_task_status(self, plan_id: str, task_id: str, status: str):
        """Update the status of a task"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return
        
        for task in plan.tasks:
            if task.task_id == task_id:
                task.status = status
                logger.info(f"📊 Task {task_id} status: {status}")
                break
    
    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """Get progress of a plan"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return {}
        
        total = len(plan.tasks)
        completed = len([t for t in plan.tasks if t.status == "completed"])
        failed = len([t for t in plan.tasks if t.status == "failed"])
        pending = len([t for t in plan.tasks if t.status == "pending"])
        
        return {
            "plan_id": plan_id,
            "goal": plan.goal,
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
            "status": plan.status
        }
