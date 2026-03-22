"""
JARVIS Automation Agent
Handles real-world automation, freelancing, web interactions, APIs
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


class AutomationType(Enum):
    """Types of automation"""
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    EMAIL_AUTOMATION = "email_automation"
    FORM_FILLING = "form_filling"
    DATA_EXTRACTION = "data_extraction"
    JOB_APPLICATION = "job_application"
    CONTENT_PUBLISHING = "content_publishing"
    SCHEDULED_TASK = "scheduled_task"


class AutomationTask:
    """An automation task"""
    def __init__(
        self,
        task_id: str,
        automation_type: AutomationType,
        description: str,
        params: Dict[str, Any],
        requires_permission: bool = True,
        risk_level: str = "medium"
    ):
        self.task_id = task_id
        self.automation_type = automation_type
        self.description = description
        self.params = params
        self.requires_permission = requires_permission
        self.risk_level = risk_level
        
        self.status = "pending"
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
        self.permission_granted = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "automation_type": self.automation_type.value,
            "description": self.description,
            "params": self.params,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }


class JobListing:
    """A job listing from freelancing platforms"""
    def __init__(
        self,
        job_id: str,
        platform: str,  # "fiverr", "upwork", "freelancer"
        title: str,
        description: str,
        budget: Dict[str, Any],
        skills: List[str],
        posted_at: str,
        url: str,
        client_info: Dict = None
    ):
        self.job_id = job_id
        self.platform = platform
        self.title = title
        self.description = description
        self.budget = budget
        self.skills = skills
        self.posted_at = posted_at
        self.url = url
        self.client_info = client_info or {}
        
        # Analysis
        self.match_score = 0.0
        self.applied = False
        self.application_date = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "platform": self.platform,
            "title": self.title,
            "description": self.description[:200] + "..." if len(self.description) > 200 else self.description,
            "budget": self.budget,
            "skills": self.skills,
            "posted_at": self.posted_at,
            "url": self.url,
            "match_score": self.match_score,
            "applied": self.applied
        }


class Proposal:
    """A generated job proposal"""
    def __init__(
        self,
        proposal_id: str,
        job_id: str,
        content: str,
        price: float,
        delivery_time: str
    ):
        self.proposal_id = proposal_id
        self.job_id = job_id
        self.content = content
        self.price = price
        self.delivery_time = delivery_time
        self.generated_at = datetime.utcnow().isoformat()
        self.sent = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "job_id": self.job_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "price": self.price,
            "delivery_time": self.delivery_time,
            "sent": self.sent
        }


class AutomationAgent(BaseAgent):
    """
    Automation Agent: Handles real-world automation tasks
    
    Capabilities:
    - Web scraping and data extraction
    - API integrations
    - Job discovery and application (Fiverr, Upwork)
    - Proposal generation
    - Automated workflows
    - Scheduled automations
    """
    
    def __init__(
        self,
        web_scraper=None,
        api_clients: Dict = None,
        llm_client=None,
        config: Dict = None
    ):
        super().__init__(AgentType.AUTOMATION, config=config)
        
        # Clients
        self.web_scraper = web_scraper
        self.api_clients = api_clients or {}
        self.llm_client = llm_client
        
        # Automation tracking
        self.pending_tasks: Dict[str, AutomationTask] = {}
        self.completed_tasks: List[AutomationTask] = []
        self.automation_handlers: Dict[str, Callable] = {}
        
        # Freelancing
        self.job_listings: Dict[str, JobListing] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.user_skills: List[str] = config.get("user_skills", []) if config else []
        self.user_portfolio: Dict = config.get("portfolio", {}) if config else {}
        
        # Automation settings
        self.auto_apply_enabled = config.get("auto_apply", False) if config else False
        self.min_match_score = config.get("min_match_score", 0.7) if config else 0.7
        self.max_daily_applications = config.get("max_daily_applications", 5) if config else 5
        
        # Register handlers
        self._register_automation_handlers()
    
    def _register_automation_handlers(self):
        """Register automation handlers"""
        self.automation_handlers[AutomationType.WEB_SCRAPING.value] = self._handle_web_scraping
        self.automation_handlers[AutomationType.API_INTEGRATION.value] = self._handle_api_call
        self.automation_handlers[AutomationType.EMAIL_AUTOMATION.value] = self._handle_email_automation
        self.automation_handlers[AutomationType.JOB_APPLICATION.value] = self._handle_job_application
        self.automation_handlers[AutomationType.DATA_EXTRACTION.value] = self._handle_data_extraction
    
    async def _initialize(self) -> bool:
        """Initialize the Automation Agent"""
        logger.info("🤖 Automation Agent initializing...")
        
        # Register message handlers
        self.register_handler(MessageType.AUTOMATION_TRIGGER, self._handle_automation_trigger)
        self.register_handler(MessageType.PERMISSION_GRANTED, self._handle_permission_granted)
        self.register_handler(MessageType.EXECUTE, self._handle_execute_request)
        
        logger.info("✅ Automation Agent initialized")
        return True
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.pending_tasks.clear()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle generic messages"""
        logger.debug(f"📥 Automation Agent received: {message.message_type}")
    
    async def _handle_automation_trigger(self, message: AgentMessage):
        """Handle automation trigger requests"""
        content = message.content
        automation_type = AutomationType(content.get("automation_type", "web_scraping"))
        description = content.get("description", "")
        params = content.get("params", {})
        
        logger.info(f"🤖 Automation triggered: {automation_type.value}")
        
        # Create automation task
        task = AutomationTask(
            task_id=f"auto_{datetime.utcnow().timestamp()}",
            automation_type=automation_type,
            description=description,
            params=params,
            requires_permission=content.get("requires_permission", True),
            risk_level=content.get("risk_level", "medium")
        )
        
        self.pending_tasks[task.task_id] = task
        
        # Request permission if needed
        if task.requires_permission:
            await self._request_permission(task, message)
            return
        
        # Execute immediately
        await self._execute_automation_task(task, message)
    
    async def _handle_permission_granted(self, message: AgentMessage):
        """Handle permission granted for automation"""
        content = message.content
        task_id = content.get("task_id")
        
        task = self.pending_tasks.get(task_id)
        if task:
            task.permission_granted = True
            await self._execute_automation_task(task, None)
    
    async def _handle_execute_request(self, message: AgentMessage):
        """Handle direct execution requests"""
        content = message.content
        action = content.get("action", "")
        params = content.get("params", {})
        
        if action == "discover_jobs":
            result = await self.discover_jobs(
                platforms=params.get("platforms", ["upwork", "fiverr"]),
                keywords=params.get("keywords", []),
                limit=params.get("limit", 10)
            )
            await self.send_response(
                original_message=message,
                content={"jobs": [j.to_dict() for j in result]}
            )
        
        elif action == "generate_proposal":
            proposal = await self.generate_proposal(
                job_id=params.get("job_id"),
                custom_info=params.get("custom_info", {})
            )
            await self.send_response(
                original_message=message,
                content={"proposal": proposal.to_dict() if proposal else None}
            )
        
        elif action == "apply_to_job":
            result = await self.apply_to_job(
                job_id=params.get("job_id"),
                proposal_id=params.get("proposal_id")
            )
            await self.send_response(
                original_message=message,
                content={"applied": result}
            )
    
    async def _request_permission(self, task: AutomationTask, original_message: AgentMessage):
        """Request permission for automation"""
        await self.send_message(
            to_agent=AgentType.DECISION,
            message_type=MessageType.PERMISSION_REQUEST,
            content={
                "task_id": task.task_id,
                "action": task.description,
                "risk_level": task.risk_level,
                "details": task.params
            },
            priority=1
        )
    
    async def _execute_automation_task(
        self,
        task: AutomationTask,
        original_message: Optional[AgentMessage]
    ):
        """Execute an automation task"""
        task.status = "running"
        
        try:
            handler = self.automation_handlers.get(task.automation_type.value)
            if handler:
                result = await handler(task.params)
                task.result = result
                task.status = "completed"
                task.completed_at = datetime.utcnow().isoformat()
                
                logger.info(f"✅ Automation completed: {task.task_id}")
                
                # Send result
                if original_message:
                    await self.send_response(
                        original_message=original_message,
                        content={"task_id": task.task_id, "result": result}
                    )
                
                # Broadcast completion
                await self.broadcast(
                    message_type=MessageType.AUTOMATION_RESULT,
                    content={
                        "task_id": task.task_id,
                        "automation_type": task.automation_type.value,
                        "status": "completed",
                        "result": result
                    }
                )
            else:
                raise ValueError(f"No handler for automation type: {task.automation_type}")
                
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"❌ Automation failed: {e}")
            
            if original_message:
                await self.send_response(
                    original_message=original_message,
                    content={"task_id": task.task_id, "error": str(e)},
                    success=False
                )
        
        # Move to completed
        self.completed_tasks.append(task)
        if task.task_id in self.pending_tasks:
            del self.pending_tasks[task.task_id]
    
    # Automation handlers
    async def _handle_web_scraping(self, params: Dict) -> Dict:
        """Handle web scraping automation"""
        url = params.get("url", "")
        selectors = params.get("selectors", {})
        
        logger.info(f"🔍 Scraping: {url}")
        
        if self.web_scraper:
            return await self.web_scraper.scrape(url, selectors)
        else:
            return {"error": "Web scraper not configured"}
    
    async def _handle_api_call(self, params: Dict) -> Dict:
        """Handle API call automation"""
        api_name = params.get("api_name", "")
        endpoint = params.get("endpoint", "")
        method = params.get("method", "GET")
        data = params.get("data", {})
        
        logger.info(f"🌐 API call: {api_name} - {endpoint}")
        
        client = self.api_clients.get(api_name)
        if client:
            return await client.call(endpoint, method, data)
        else:
            return {"error": f"API client not found: {api_name}"}
    
    async def _handle_email_automation(self, params: Dict) -> Dict:
        """Handle email automation"""
        # Forward to Communication Agent
        await self.send_message(
            to_agent=AgentType.COMMUNICATION,
            message_type=MessageType.REQUEST,
            content={
                "type": "send_email",
                **params
            }
        )
        return {"status": "forwarded"}
    
    async def _handle_job_application(self, params: Dict) -> Dict:
        """Handle job application automation"""
        job_id = params.get("job_id", "")
        proposal_id = params.get("proposal_id", "")
        
        result = await self.apply_to_job(job_id, proposal_id)
        return {"applied": result}
    
    async def _handle_data_extraction(self, params: Dict) -> Dict:
        """Handle data extraction automation"""
        source = params.get("source", "")
        query = params.get("query", "")
        
        logger.info(f"📊 Extracting data from: {source}")
        
        # Implementation depends on data source
        return {"source": source, "query": query, "status": "extracted"}
    
    # Freelancing automation methods
    async def discover_jobs(
        self,
        platforms: List[str] = None,
        keywords: List[str] = None,
        limit: int = 10
    ) -> List[JobListing]:
        """Discover jobs from freelancing platforms"""
        platforms = platforms or ["upwork", "fiverr"]
        keywords = keywords or self.user_skills
        
        logger.info(f"🔍 Discovering jobs on {platforms} for {keywords}")
        
        all_jobs = []
        
        for platform in platforms:
            if platform == "upwork":
                jobs = await self._scrape_upwork(keywords, limit)
            elif platform == "fiverr":
                jobs = await self._scrape_fiverr(keywords, limit)
            else:
                continue
            
            # Calculate match scores
            for job in jobs:
                job.match_score = self._calculate_match_score(job, keywords)
            
            # Sort by match score
            jobs.sort(key=lambda j: j.match_score, reverse=True)
            
            all_jobs.extend(jobs)
        
        # Store jobs
        for job in all_jobs:
            self.job_listings[job.job_id] = job
        
        logger.info(f"✅ Discovered {len(all_jobs)} jobs")
        
        return all_jobs[:limit]
    
    async def _scrape_upwork(self, keywords: List[str], limit: int) -> List[JobListing]:
        """Scrape Upwork using their public RSS feed"""
        import httpx
        import xml.etree.ElementTree as ET
        
        jobs = []
        query = "+".join(keywords) if keywords else "developer"
        url = f"https://www.upwork.com/ab/feed/jobs/rss?q={query}"
        
        logger.info(f"🌐 Fetching live Upwork RSS: {url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                root = ET.fromstring(response.text)
                
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title").text if item.find("title") is not None else "Unknown Job"
                    desc = item.find("description").text if item.find("description") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else url
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else datetime.utcnow().isoformat()
                    
                    job = JobListing(
                        job_id=f"upwork_{datetime.utcnow().timestamp()}_{len(jobs)}",
                        platform="upwork",
                        title=title,
                        description=desc[:500],
                        budget={"type": "unknown", "amount": 0},
                        skills=keywords,
                        posted_at=pub_date,
                        url=link,
                        client_info={"rating": 0, "jobs_posted": 0}
                    )
                    jobs.append(job)
        except Exception as e:
            logger.error(f"Failed to scrape real Upwork jobs: {e}")
            
        return jobs
    
    async def _scrape_fiverr(self, keywords: List[str], limit: int) -> List[JobListing]:
        """Scrape Fiverr for job listings (Buyer Requests)"""
        # Fiverr aggressively blocks httpx and requires Puppeteer stealth.
        # Fallback to empty list to prevent bans, driving traffic entirely through Upwork RSS.
        logger.warning("Fiverr scraping bypassed in v3.0 to prevent server shadow bans without Stealth Puppeteer.")
        return []
    
    def _calculate_match_score(self, job: JobListing, user_skills: List[str]) -> float:
        """Calculate match score between job and user skills"""
        if not user_skills:
            return 0.5
        
        # Skill overlap
        job_skills = set(s.lower() for s in job.skills)
        user_skills_set = set(s.lower() for s in user_skills)
        
        if not job_skills:
            return 0.5
        
        overlap = len(job_skills & user_skills_set)
        skill_score = overlap / len(job_skills)
        
        # Budget attractiveness (higher budget = better)
        budget_score = 0.5
        if job.budget.get("amount"):
            budget_score = min(job.budget["amount"] / 1000, 1.0)
        elif job.budget.get("max"):
            budget_score = min(job.budget["max"] / 1000, 1.0)
        
        # Client rating
        client_score = job.client_info.get("rating", 3) / 5
        
        # Combined score
        return (skill_score * 0.5) + (budget_score * 0.3) + (client_score * 0.2)
    
    async def generate_proposal(
        self,
        job_id: str,
        custom_info: Dict = None
    ) -> Optional[Proposal]:
        """Generate a job proposal using LLM"""
        job = self.job_listings.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return None
        
        logger.info(f"📝 Generating proposal for: {job.title}")
        
        # Build prompt for LLM
        prompt = self._build_proposal_prompt(job, custom_info)
        
        if self.llm_client:
            try:
                response = await self.llm_client.generate(prompt)
                
                proposal_content = response.get("content", "")
                price = response.get("price", job.budget.get("amount", 100))
                delivery_time = response.get("delivery_time", "3 days")
                
                proposal = Proposal(
                    proposal_id=f"prop_{datetime.utcnow().timestamp()}",
                    job_id=job_id,
                    content=proposal_content,
                    price=price,
                    delivery_time=delivery_time
                )
                
                self.proposals[proposal.proposal_id] = proposal
                
                logger.info(f"✅ Proposal generated: {proposal.proposal_id}")
                
                return proposal
                
            except Exception as e:
                logger.error(f"❌ Proposal generation failed: {e}")
                return None
        else:
            # Default proposal
            proposal = Proposal(
                proposal_id=f"prop_{datetime.utcnow().timestamp()}",
                job_id=job_id,
                content=f"Hi, I'm interested in your project '{job.title}'. I have extensive experience in {', '.join(job.skills)} and can deliver high-quality results.",
                price=job.budget.get("amount", 100),
                delivery_time="3 days"
            )
            self.proposals[proposal.proposal_id] = proposal
            return proposal
    
    def _build_proposal_prompt(self, job: JobListing, custom_info: Dict = None) -> str:
        """Build prompt for proposal generation"""
        user_skills = ", ".join(self.user_skills)
        portfolio = json.dumps(self.user_portfolio, indent=2)
        
        return f"""You are an expert freelancer applying for jobs. Generate a compelling proposal for the following job:

Job Title: {job.title}
Job Description: {job.description}
Required Skills: {', '.join(job.skills)}
Budget: {job.budget}

Your Skills: {user_skills}
Your Portfolio: {portfolio}

Additional Info: {json.dumps(custom_info or {})}

Generate a proposal that:
1. Shows understanding of the project
2. Highlights relevant experience
3. Proposes a fair price and timeline
4. Includes a call to action

Return JSON:
{{
    "content": "<proposal text>",
    "price": <suggested price>,
    "delivery_time": "<time estimate>"
}}"""
    
    async def apply_to_job(self, job_id: str, proposal_id: str = None) -> bool:
        """Apply to a job with a proposal"""
        job = self.job_listings.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        # Get or generate proposal
        proposal = self.proposals.get(proposal_id) if proposal_id else None
        if not proposal:
            proposal = await self.generate_proposal(job_id)
        
        if not proposal:
            return False
        
        logger.info(f"📤 Applying to job: {job.title}")
        
        # This would integrate with platform APIs
        # For now, simulate the application
        
        # Request permission for auto-apply
        if not self.auto_apply_enabled:
            await self.send_message(
                to_agent=AgentType.DECISION,
                message_type=MessageType.PERMISSION_REQUEST,
                content={
                    "action": f"Apply to job: {job.title}",
                    "risk_level": "medium",
                    "details": {
                        "job": job.to_dict(),
                        "proposal": proposal.to_dict()
                    }
                }
            )
            return False
        
        # Mark as applied
        job.applied = True
        job.application_date = datetime.utcnow().isoformat()
        proposal.sent = True
        
        logger.info(f"✅ Applied to job: {job.title}")
        
        # Notify user
        await self.send_message(
            to_agent=AgentType.COMMUNICATION,
            message_type=MessageType.REQUEST,
            content={
                "type": "notification",
                "title": "Job Application Sent",
                "message": f"Applied to '{job.title}' on {job.platform}",
                "priority": "normal"
            }
        )
        
        return True
    
    async def run_money_automation(self) -> Dict[str, Any]:
        """Run the complete 'Make me money today' automation"""
        logger.info("💰 Running money automation...")
        
        results = {
            "jobs_discovered": 0,
            "proposals_generated": 0,
            "applications_sent": 0,
            "errors": []
        }
        
        try:
            # 1. Discover jobs
            jobs = await self.discover_jobs(
                platforms=["upwork", "fiverr"],
                limit=self.max_daily_applications * 2
            )
            results["jobs_discovered"] = len(jobs)
            
            # 2. Filter high-match jobs
            high_match_jobs = [j for j in jobs if j.match_score >= self.min_match_score]
            
            # 3. Generate proposals
            for job in high_match_jobs[:self.max_daily_applications]:
                proposal = await self.generate_proposal(job.job_id)
                if proposal:
                    results["proposals_generated"] += 1
            
            # 4. Apply to jobs (if auto-apply enabled)
            if self.auto_apply_enabled:
                for job in high_match_jobs[:self.max_daily_applications]:
                    success = await self.apply_to_job(job.job_id)
                    if success:
                        results["applications_sent"] += 1
            
        except Exception as e:
            logger.error(f"❌ Money automation error: {e}")
            results["errors"].append(str(e))
        
        logger.info(f"💰 Money automation complete: {results}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        return {
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "job_listings": len(self.job_listings),
            "proposals": len(self.proposals),
            "applications_sent": len([j for j in self.job_listings.values() if j.applied]),
            "auto_apply_enabled": self.auto_apply_enabled
        }
