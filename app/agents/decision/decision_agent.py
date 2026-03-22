"""
JARVIS Decision Agent
Makes intelligent decisions, evaluates options, manages permissions
Acts as the "brain" for critical choices
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


class RiskLevel(Enum):
    """Risk levels for decisions"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionType(Enum):
    """Types of decisions"""
    AUTOMATIC = "automatic"  # No user approval needed
    RECOMMENDED = "recommended"  # User approval recommended
    REQUIRED = "required"  # User approval required
    EMERGENCY = "emergency"  # Override possible


class DecisionOutcome(Enum):
    """Possible decision outcomes"""
    APPROVED = "approved"
    DENIED = "denied"
    DEFERRED = "deferred"
    ESCALATED = "escalated"
    PENDING = "pending"


class Decision:
    """Represents a decision"""
    def __init__(
        self,
        decision_id: str,
        decision_type: DecisionType,
        risk_level: RiskLevel,
        description: str,
        options: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
        timeout_seconds: int = 300
    ):
        self.decision_id = decision_id
        self.decision_type = decision_type
        self.risk_level = risk_level
        self.description = description
        self.options = options
        self.context = context or {}
        self.timeout_seconds = timeout_seconds
        
        self.outcome = DecisionOutcome.PENDING
        self.selected_option = None
        self.reasoning = ""
        self.approved_by = None
        self.created_at = datetime.utcnow().isoformat()
        self.decided_at = None
        self.expires_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "options": self.options,
            "context": self.context,
            "outcome": self.outcome.value,
            "selected_option": self.selected_option,
            "reasoning": self.reasoning,
            "approved_by": self.approved_by,
            "created_at": self.created_at,
            "decided_at": self.decided_at
        }


class PermissionRequest:
    """Permission request for user approval"""
    def __init__(
        self,
        request_id: str,
        action: str,
        risk_level: RiskLevel,
        details: Dict[str, Any],
        requester: str
    ):
        self.request_id = request_id
        self.action = action
        self.risk_level = risk_level
        self.details = details
        self.requester = requester
        self.status = "pending"  # pending, approved, denied, expired
        self.user_response = None
        self.created_at = datetime.utcnow().isoformat()
        self.responded_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "risk_level": self.risk_level.value,
            "details": self.details,
            "requester": self.requester,
            "status": self.status,
            "user_response": self.user_response,
            "created_at": self.created_at,
            "responded_at": self.responded_at
        }


class DecisionAgent(BaseAgent):
    """
    Decision Agent: Makes intelligent decisions and manages permissions
    
    Capabilities:
    - Evaluate options and make decisions
    - Assess risk levels
    - Request user permissions
    - Learn from past decisions
    - Handle emergency overrides
    """
    
    def __init__(self, llm_client=None, config: Dict = None):
        super().__init__(AgentType.DECISION, config=config)
        self.llm_client = llm_client
        
        # Decision tracking
        self.pending_decisions: Dict[str, Decision] = {}
        self.decision_history: List[Decision] = []
        self.pending_permissions: Dict[str, PermissionRequest] = {}
        
        # Decision policies
        self.auto_approve_risk_levels = [RiskLevel.NONE, RiskLevel.LOW]
        self.user_approval_risk_levels = [RiskLevel.MEDIUM, RiskLevel.HIGH]
        self.emergency_override_enabled = config.get("emergency_override", True) if config else True
        
        # Learning
        self.decision_patterns: Dict[str, List[Decision]] = {}
        self.user_preferences: Dict[str, Any] = {}
        
        # Callbacks
        self.on_permission_request: Optional[Callable] = None
        self.on_decision_made: Optional[Callable] = None
    
    async def _initialize(self) -> bool:
        """Initialize the Decision Agent"""
        logger.info("🧠 Decision Agent initializing...")
        
        # Register message handlers
        self.register_handler(MessageType.PERMISSION_REQUEST, self._handle_permission_request)
        self.register_handler(MessageType.REQUEST, self._handle_decision_request)
        self.register_handler(MessageType.REFLECT, self._handle_reflection)
        
        logger.info("✅ Decision Agent initialized")
        return True
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.pending_decisions.clear()
        self.pending_permissions.clear()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle generic messages"""
        logger.info(f"📥 Decision Agent received: {message.message_type}")
    
    async def _handle_permission_request(self, message: AgentMessage):
        """Handle permission requests from other agents"""
        content = message.content
        action = content.get("action", "")
        risk_level = RiskLevel(content.get("risk_level", "medium"))
        details = content.get("details", {})
        
        logger.info(f"🔐 Permission request: {action} (Risk: {risk_level.value})")
        
        # Create permission request
        request_id = f"perm_{datetime.utcnow().timestamp()}"
        permission = PermissionRequest(
            request_id=request_id,
            action=action,
            risk_level=risk_level,
            details=details,
            requester=message.from_agent
        )
        
        self.pending_permissions[request_id] = permission
        
        # Auto-approve low risk actions
        if risk_level in self.auto_approve_risk_levels:
            await self._auto_approve_permission(request_id)
            return
        
        # Request user approval for medium/high risk
        if risk_level in self.user_approval_risk_levels:
            await self._request_user_permission(request_id, message)
            return
        
        # Critical actions require explicit user approval
        if risk_level == RiskLevel.CRITICAL:
            await self._request_user_permission(request_id, message, urgent=True)
            return
    
    async def _handle_decision_request(self, message: AgentMessage):
        """Handle decision requests"""
        content = message.content
        decision_type = DecisionType(content.get("decision_type", "automatic"))
        description = content.get("description", "")
        options = content.get("options", [])
        context = content.get("context", {})
        
        logger.info(f"🤔 Decision request: {description}")
        
        # Make the decision
        decision = await self.make_decision(
            decision_type=decision_type,
            description=description,
            options=options,
            context=context
        )
        
        # Send response
        await self.send_response(
            original_message=message,
            content={
                "decision_id": decision.decision_id,
                "outcome": decision.outcome.value,
                "selected_option": decision.selected_option,
                "reasoning": decision.reasoning
            }
        )
    
    async def _handle_reflection(self, message: AgentMessage):
        """Handle reflection messages for learning"""
        content = message.content
        result = content.get("result", {})
        success = content.get("success", False)
        
        logger.info(f"🪞 Reflection received: success={success}")
        
        # Learn from the reflection
        await self._learn_from_reflection(result, success)
    
    async def make_decision(
        self,
        decision_type: DecisionType,
        description: str,
        options: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
        risk_level: RiskLevel = None
    ) -> Decision:
        """Make a decision"""
        
        # Assess risk if not provided
        if risk_level is None:
            risk_level = await self._assess_risk(description, options, context)
        
        # Create decision
        decision_id = f"dec_{datetime.utcnow().timestamp()}"
        decision = Decision(
            decision_id=decision_id,
            decision_type=decision_type,
            risk_level=risk_level,
            description=description,
            options=options,
            context=context or {}
        )
        
        self.pending_decisions[decision_id] = decision
        
        # Auto-decide for low risk
        if risk_level in self.auto_approve_risk_levels and decision_type == DecisionType.AUTOMATIC:
            await self._auto_decide(decision)
            return decision
        
        # Use LLM for complex decisions
        if self.llm_client and len(options) > 1:
            await self._llm_decide(decision)
        else:
            # Simple decision: choose first option
            await self._finalize_decision(
                decision,
                DecisionOutcome.APPROVED,
                options[0] if options else None,
                "Default selection"
            )
        
        # Store in history
        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)
        
        # Track pattern
        pattern_key = self._get_pattern_key(description)
        if pattern_key not in self.decision_patterns:
            self.decision_patterns[pattern_key] = []
        self.decision_patterns[pattern_key].append(decision)
        
        # Notify callback
        if self.on_decision_made:
            await self.on_decision_made(decision)
        
        logger.info(f"✅ Decision made: {decision.outcome.value}")
        
        return decision
    
    async def _assess_risk(
        self,
        description: str,
        options: List[Dict],
        context: Dict
    ) -> RiskLevel:
        """Assess the risk level of a decision"""
        risk_score = 0
        
        # Check for keywords indicating risk
        high_risk_keywords = ["delete", "remove", "payment", "money", "password", "private"]
        medium_risk_keywords = ["send", "post", "share", "modify", "update"]
        
        desc_lower = description.lower()
        
        for keyword in high_risk_keywords:
            if keyword in desc_lower:
                risk_score += 3
        
        for keyword in medium_risk_keywords:
            if keyword in desc_lower:
                risk_score += 1
        
        # Check context for additional risk factors
        if context.get("financial_impact", False):
            risk_score += 2
        if context.get("irreversible", False):
            risk_score += 2
        if context.get("public_visibility", False):
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _auto_decide(self, decision: Decision):
        """Make an automatic decision for low-risk scenarios"""
        # Select the best option based on simple heuristics
        best_option = None
        best_score = -1
        
        for option in decision.options:
            score = option.get("confidence", 0.5)
            if score > best_score:
                best_score = score
                best_option = option
        
        await self._finalize_decision(
            decision,
            DecisionOutcome.APPROVED,
            best_option or (decision.options[0] if decision.options else None),
            "Auto-approved: low risk"
        )
    
    async def _llm_decide(self, decision: Decision):
        """Use LLM to make a decision"""
        if not self.llm_client:
            return
        
        # Build prompt for LLM
        prompt = self._build_decision_prompt(decision)
        
        # Get LLM decision
        llm_response = await self.llm_client.decide(prompt)
        
        # Parse response
        selected_index = llm_response.get("selected_option", 0)
        reasoning = llm_response.get("reasoning", "")
        
        selected_option = decision.options[selected_index] if 0 <= selected_index < len(decision.options) else None
        
        await self._finalize_decision(
            decision,
            DecisionOutcome.APPROVED,
            selected_option,
            reasoning
        )
    
    def _build_decision_prompt(self, decision: Decision) -> str:
        """Build a prompt for LLM decision making"""
        options_text = "\n".join([
            f"{i}. {opt.get('description', 'Option ' + str(i))} (Confidence: {opt.get('confidence', 0.5)})"
            for i, opt in enumerate(decision.options)
        ])
        
        return f"""You are JARVIS Decision Agent. Analyze the following decision:

Description: {decision.description}
Risk Level: {decision.risk_level.value}
Context: {json.dumps(decision.context, indent=2)}

Options:
{options_text}

Select the best option and provide reasoning. Return JSON:
{{
    "selected_option": <index>,
    "reasoning": "<your reasoning>"
}}"""
    
    async def _finalize_decision(
        self,
        decision: Decision,
        outcome: DecisionOutcome,
        selected_option: Dict,
        reasoning: str
    ):
        """Finalize a decision"""
        decision.outcome = outcome
        decision.selected_option = selected_option
        decision.reasoning = reasoning
        decision.decided_at = datetime.utcnow().isoformat()
        
        del self.pending_decisions[decision.decision_id]
    
    async def _auto_approve_permission(self, request_id: str):
        """Auto-approve a permission request"""
        permission = self.pending_permissions.get(request_id)
        if not permission:
            return
        
        permission.status = "approved"
        permission.responded_at = datetime.utcnow().isoformat()
        
        # Notify the requester
        await self.send_message(
            to_agent=AgentType(permission.requester),
            message_type=MessageType.PERMISSION_GRANTED,
            content={
                "request_id": request_id,
                "action": permission.action,
                "auto_approved": True
            }
        )
        
        logger.info(f"✅ Auto-approved: {permission.action}")
    
    async def _request_user_permission(
        self,
        request_id: str,
        original_message: AgentMessage,
        urgent: bool = False
    ):
        """Request permission from user"""
        permission = self.pending_permissions.get(request_id)
        if not permission:
            return
        
        # Notify via Communication Agent
        await self.send_message(
            to_agent=AgentType.COMMUNICATION,
            message_type=MessageType.REQUEST,
            content={
                "type": "permission_request",
                "urgent": urgent,
                "permission": permission.to_dict()
            },
            priority=1 if urgent else 3
        )
        
        # Notify callback
        if self.on_permission_request:
            await self.on_permission_request(permission)
        
        logger.info(f"⏳ User permission requested: {permission.action}")
    
    async def respond_to_permission(
        self,
        request_id: str,
        approved: bool,
        user_response: str = None
    ):
        """Handle user response to permission request"""
        permission = self.pending_permissions.get(request_id)
        if not permission:
            logger.error(f"Permission request not found: {request_id}")
            return False
        
        permission.status = "approved" if approved else "denied"
        permission.user_response = user_response
        permission.responded_at = datetime.utcnow().isoformat()
        
        # Notify the requester
        message_type = MessageType.PERMISSION_GRANTED if approved else MessageType.PERMISSION_DENIED
        
        await self.send_message(
            to_agent=AgentType(permission.requester),
            message_type=message_type,
            content={
                "request_id": request_id,
                "action": permission.action,
                "user_response": user_response
            }
        )
        
        # Learn from user decision
        await self._learn_from_permission(permission)
        
        logger.info(f"{'✅' if approved else '❌'} Permission {permission.status}: {permission.action}")
        
        return True
    
    async def _learn_from_permission(self, permission: PermissionRequest):
        """Learn from user permission decisions"""
        action_pattern = self._get_pattern_key(permission.action)
        
        # Track user preference for this action type
        if action_pattern not in self.user_preferences:
            self.user_preferences[action_pattern] = {
                "approved": 0,
                "denied": 0,
                "pattern": action_pattern
            }
        
        if permission.status == "approved":
            self.user_preferences[action_pattern]["approved"] += 1
        else:
            self.user_preferences[action_pattern]["denied"] += 1
    
    async def _learn_from_reflection(self, result: Dict, success: bool):
        """Learn from reflection results"""
        # Update decision quality metrics
        pass  # Implementation depends on specific learning strategy
    
    def _get_pattern_key(self, text: str) -> str:
        """Extract a pattern key from text"""
        # Simple pattern extraction - can be improved
        words = text.lower().split()
        significant_words = [w for w in words if len(w) > 3]
        return "_".join(significant_words[:3]) if significant_words else "general"
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics"""
        total = len(self.decision_history)
        approved = len([d for d in self.decision_history if d.outcome == DecisionOutcome.APPROVED])
        denied = len([d for d in self.decision_history if d.outcome == DecisionOutcome.DENIED])
        
        return {
            "total_decisions": total,
            "approved": approved,
            "denied": denied,
            "pending": len(self.pending_decisions),
            "approval_rate": (approved / total * 100) if total > 0 else 0,
            "user_preferences_count": len(self.user_preferences)
        }
    
    def get_pending_permissions(self) -> List[Dict]:
        """Get all pending permission requests"""
        return [p.to_dict() for p in self.pending_permissions.values() if p.status == "pending"]
