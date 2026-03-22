"""
JARVIS Execution Agent
Executes tasks, manages workflows, handles real-world actions
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

from app.core.base_agent import BaseAgent, AgentState
from app.core.message_bus import (
    AgentMessage, MessageType, AgentType, MessageBuilder, message_bus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution statuses"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class Task:
    """Represents an executable task"""
    def __init__(
        self,
        task_id: str,
        task_type: str,
        description: str,
        params: Dict[str, Any] = None,
        priority: int = 5,
        max_retries: int = 3,
        timeout_seconds: int = 300
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.params = params or {}
        self.priority = priority
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.retry_count = 0
        
        self.created_at = datetime.utcnow().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        
        self.subtasks: List[str] = []
        self.dependencies: List[str] = []
        self.assigned_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "params": self.params,
            "priority": self.priority,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "subtasks": self.subtasks,
            "dependencies": self.dependencies
        }


class Workflow:
    """A workflow of related tasks"""
    def __init__(
        self,
        workflow_id: str,
        name: str,
        description: str = ""
    ):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
    
    def add_task(self, task: Task):
        """Add a task to the workflow"""
        self.tasks[task.task_id] = task
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)"""
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_met = all(
                    self.tasks.get(dep) and self.tasks[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_met:
                    ready.append(task)
        return ready
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "task_count": len(self.tasks),
            "completed_count": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }


class ExecutionAgent(BaseAgent):
    """
    Execution Agent: Executes tasks and manages workflows
    
    Capabilities:
    - Execute various types of tasks
    - Manage task workflows
    - Handle retries and failures
    - Execute in parallel where possible
    - Report progress and results
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(AgentType.EXECUTION, config=config)
        
        # Task management
        self.pending_tasks: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.workflows: Dict[str, Workflow] = {}
        
        # Execution settings
        self.max_concurrent_tasks = config.get("max_concurrent", 5) if config else 5
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        
        # Register built-in handlers
        self._register_builtin_handlers()
    
    def _register_builtin_handlers(self):
        """Register built-in task handlers"""
        self.task_handlers["shell"] = self._handle_shell_task
        self.task_handlers["api_call"] = self._handle_api_task
        self.task_handlers["web_scrape"] = self._handle_scrape_task
        self.task_handlers["email"] = self._handle_email_task
        self.task_handlers["notification"] = self._handle_notification_task
        self.task_handlers["wait"] = self._handle_wait_task
        self.task_handlers["callback"] = self._handle_callback_task
    
    async def _initialize(self) -> bool:
        """Initialize the Execution Agent"""
        logger.info("⚡ Execution Agent initializing...")
        
        # Register message handlers
        self.register_handler(MessageType.EXECUTE, self._handle_execute_message)
        self.register_handler(MessageType.TASK_ASSIGN, self._handle_task_assign)
        self.register_handler(MessageType.PLAN, self._handle_plan_message)
        
        # Start task processor
        asyncio.create_task(self._task_processor())
        
        logger.info("✅ Execution Agent initialized")
        return True
    
    async def _cleanup(self):
        """Cleanup resources"""
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        self.active_tasks.clear()
        self.workflows.clear()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle generic messages"""
        logger.info(f"📥 Execution Agent received: {message.message_type}")
    
    async def _handle_execute_message(self, message: AgentMessage):
        """Handle direct execution requests"""
        content = message.content
        action = content.get("action", "")
        params = content.get("params", {})
        
        logger.info(f"⚡ Execute request: {action}")
        
        # Create and execute task
        task = Task(
            task_id=f"task_{datetime.utcnow().timestamp()}",
            task_type=action,
            description=f"Execute {action}",
            params=params,
            priority=content.get("priority", 5)
        )
        
        result = await self.execute_task(task)
        
        # Send response
        await self.send_response(
            original_message=message,
            content={
                "task_id": task.task_id,
                "status": task.status.value,
                "result": result,
                "error": task.error
            },
            success=task.status == TaskStatus.COMPLETED
        )
    
    async def _handle_task_assign(self, message: AgentMessage):
        """Handle task assignment"""
        content = message.content
        task_data = content.get("task", {})
        
        task = Task(
            task_id=task_data.get("task_id", f"task_{datetime.utcnow().timestamp()}"),
            task_type=task_data.get("task_type", "unknown"),
            description=task_data.get("description", ""),
            params=task_data.get("params", {}),
            priority=task_data.get("priority", 5),
            max_retries=task_data.get("max_retries", 3)
        )
        
        # Queue the task
        await self.queue_task(task)
        
        # Send acknowledgment
        await self.send_response(
            original_message=message,
            content={
                "task_id": task.task_id,
                "status": "queued"
            }
        )
    
    async def _handle_plan_message(self, message: AgentMessage):
        """Handle plan execution"""
        content = message.content
        plan = content.get("plan", [])
        goal = content.get("goal", "")
        
        logger.info(f"📋 Executing plan for: {goal}")
        
        # Create workflow
        workflow_id = f"wf_{datetime.utcnow().timestamp()}"
        workflow = Workflow(workflow_id, goal)
        
        # Add tasks from plan
        for i, task_data in enumerate(plan):
            task = Task(
                task_id=task_data.get("id", f"{workflow_id}_task_{i}"),
                task_type=task_data.get("type", "execute"),
                description=task_data.get("description", ""),
                params=task_data.get("params", {}),
                priority=task_data.get("priority", 5),
                dependencies=task_data.get("dependencies", [])
            )
            workflow.add_task(task)
        
        self.workflows[workflow_id] = workflow
        
        # Execute workflow
        asyncio.create_task(self._execute_workflow(workflow_id))
        
        # Send response
        await self.send_response(
            original_message=message,
            content={
                "workflow_id": workflow_id,
                "task_count": len(workflow.tasks),
                "status": "started"
            }
        )
    
    async def _task_processor(self):
        """Background task processor"""
        while self.running:
            try:
                # Check if we can run more tasks
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    # Get next task from queue
                    try:
                        priority, task = await asyncio.wait_for(
                            self.pending_tasks.get(),
                            timeout=1.0
                        )
                        
                        # Start task execution
                        task.status = TaskStatus.RUNNING
                        task.started_at = datetime.utcnow().isoformat()
                        self.active_tasks[task.task_id] = task
                        
                        task_coro = self._execute_task_with_retry(task)
                        self.running_tasks[task.task_id] = asyncio.create_task(task_coro)
                        
                    except asyncio.TimeoutError:
                        pass
                
                # Clean up completed tasks
                completed = []
                for task_id, task in self.running_tasks.items():
                    if task.done():
                        completed.append(task_id)
                
                for task_id in completed:
                    del self.running_tasks[task_id]
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Task processor error: {e}")
                await asyncio.sleep(1)
    
    async def queue_task(self, task: Task):
        """Queue a task for execution"""
        task.status = TaskStatus.QUEUED
        await self.pending_tasks.put((task.priority, task))
        logger.info(f"📥 Task queued: {task.description}")
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a single task immediately"""
        return await self._execute_task_with_retry(task)
    
    async def _execute_task_with_retry(self, task: Task) -> Any:
        """Execute a task with retry logic"""
        while task.retry_count <= task.max_retries:
            try:
                result = await self._execute_single_task(task)
                
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.utcnow().isoformat()
                
                # Move to completed
                if task.task_id in self.active_tasks:
                    self.completed_tasks.append(task)
                    del self.active_tasks[task.task_id]
                
                logger.info(f"✅ Task completed: {task.description}")
                
                # Broadcast completion
                await self.broadcast(
                    message_type=MessageType.TASK_COMPLETE,
                    content={
                        "task_id": task.task_id,
                        "result": result
                    }
                )
                
                return result
                
            except Exception as e:
                task.retry_count += 1
                task.error = str(e)
                
                logger.error(f"❌ Task failed (attempt {task.retry_count}): {e}")
                
                if task.retry_count <= task.max_retries:
                    task.status = TaskStatus.RETRYING
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                else:
                    task.status = TaskStatus.FAILED
                    
                    # Broadcast failure
                    await self.broadcast(
                        message_type=MessageType.TASK_FAILED,
                        content={
                            "task_id": task.task_id,
                            "error": str(e)
                        }
                    )
                    
                    raise
        
        return None
    
    async def _execute_single_task(self, task: Task) -> Any:
        """Execute a single task"""
        handler = self.task_handlers.get(task.task_type)
        
        if not handler:
            raise ValueError(f"No handler for task type: {task.task_type}")
        
        # Execute with timeout
        return await asyncio.wait_for(
            handler(task),
            timeout=task.timeout_seconds
        )
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return
        
        workflow.status = TaskStatus.RUNNING
        
        while workflow.status == TaskStatus.RUNNING:
            # Get ready tasks
            ready_tasks = workflow.get_ready_tasks()
            
            if not ready_tasks:
                # Check if all tasks are complete
                all_complete = all(
                    t.status == TaskStatus.COMPLETED
                    for t in workflow.tasks.values()
                )
                
                if all_complete:
                    workflow.status = TaskStatus.COMPLETED
                    workflow.completed_at = datetime.utcnow().isoformat()
                    logger.info(f"✅ Workflow completed: {workflow.name}")
                else:
                    # Check for failures
                    any_failed = any(
                        t.status == TaskStatus.FAILED
                        for t in workflow.tasks.values()
                    )
                    
                    if any_failed:
                        workflow.status = TaskStatus.FAILED
                        logger.error(f"❌ Workflow failed: {workflow.name}")
                
                break
            
            # Execute ready tasks
            for task in ready_tasks:
                await self.queue_task(task)
            
            await asyncio.sleep(0.5)
    
    # Task handlers
    async def _handle_shell_task(self, task: Task) -> Any:
        """Handle shell command tasks"""
        import subprocess
        
        command = task.params.get("command", "")
        logger.info(f"🐚 Shell: {command}")
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    async def _handle_api_task(self, task: Task) -> Any:
        """Handle API call tasks"""
        import aiohttp
        
        url = task.params.get("url", "")
        method = task.params.get("method", "GET")
        headers = task.params.get("headers", {})
        data = task.params.get("data")
        
        logger.info(f"🌐 API: {method} {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data
            ) as response:
                return {
                    "status": response.status,
                    "data": await response.json() if response.content_type == "application/json" else await response.text()
                }
    
    async def _handle_scrape_task(self, task: Task) -> Any:
        """Handle web scraping tasks"""
        import aiohttp
        from bs4 import BeautifulSoup
        
        url = task.params.get("url", "")
        selector = task.params.get("selector", "")
        
        logger.info(f"🔍 Scrape: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                if selector:
                    elements = soup.select(selector)
                    return [e.get_text(strip=True) for e in elements]
                else:
                    return soup.get_text(strip=True)
    
    async def _handle_email_task(self, task: Task) -> Any:
        """Handle email tasks"""
        # This would integrate with the Communication Agent
        logger.info(f"📧 Email task: {task.params.get('subject', '')}")
        
        # Forward to Communication Agent
        await self.send_message(
            to_agent=AgentType.COMMUNICATION,
            message_type=MessageType.REQUEST,
            content={
                "type": "send_email",
                **task.params
            }
        )
        
        return {"status": "forwarded_to_communication"}
    
    async def _handle_notification_task(self, task: Task) -> Any:
        """Handle notification tasks"""
        logger.info(f"🔔 Notification: {task.params.get('message', '')}")
        
        # Forward to Communication Agent
        await self.send_message(
            to_agent=AgentType.COMMUNICATION,
            message_type=MessageType.REQUEST,
            content={
                "type": "notification",
                **task.params
            }
        )
        
        return {"status": "notified"}
    
    async def _handle_wait_task(self, task: Task) -> Any:
        """Handle wait tasks"""
        duration = task.params.get("duration", 1)
        logger.info(f"⏳ Wait: {duration}s")
        await asyncio.sleep(duration)
        return {"waited": duration}
    
    async def _handle_callback_task(self, task: Task) -> Any:
        """Handle callback tasks"""
        callback_url = task.params.get("callback_url", "")
        payload = task.params.get("payload", {})
        
        logger.info(f"📞 Callback: {callback_url}")
        
        # Make callback request
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(callback_url, json=payload) as response:
                return {
                    "status": response.status,
                    "response": await response.text()
                }
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a custom task handler"""
        self.task_handlers[task_type] = handler
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a task"""
        task = self.active_tasks.get(task_id)
        if not task:
            task = next((t for t in self.completed_tasks if t.task_id == task_id), None)
        
        return task.to_dict() if task else None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get status of a workflow"""
        workflow = self.workflows.get(workflow_id)
        return workflow.to_dict() if workflow else None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "running_workflows": len([w for w in self.workflows.values() if w.status == TaskStatus.RUNNING]),
            "completed_workflows": len([w for w in self.workflows.values() if w.status == TaskStatus.COMPLETED]),
            "pending_queue_size": self.pending_tasks.qsize()
        }
