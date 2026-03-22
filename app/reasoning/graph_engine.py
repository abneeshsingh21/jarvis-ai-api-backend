import operator
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from app.core.llm_client import LLMClient
from app.core.message_bus import AgentType, MessageType

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """The state of the ReAct reasoning graph."""
    goal: str
    context: Dict[str, Any]
    
    thoughts: Annotated[List[Dict[str, Any]], operator.add]
    plan: List[Dict[str, Any]]
    
    # Store real executed results
    executed_actions: Annotated[List[Dict[str, Any]], operator.add]
    reflection: Dict[str, Any]
    
    iterations: int
    max_iterations: int
    error: Optional[str]


class LangGraphEngine:
    """
    Production-grade LangGraph Orchestrator replacing the fragile custom ReActLoop.
    """
    def __init__(self, orchestrator, max_iterations: int = 5):
        self.orchestrator = orchestrator # Reference to the main orchestrator for real execution
        self.max_iterations = max_iterations
        self.llm = LLMClient()
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("think", self.node_think)
        workflow.add_node("plan", self.node_plan)
        workflow.add_node("execute", self.node_execute)
        workflow.add_node("reflect", self.node_reflect)
        
        # Add edges
        workflow.add_edge(START, "think")
        workflow.add_edge("think", "plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "reflect")
        
        # Add conditional routing after reflection
        workflow.add_conditional_edges(
            "reflect",
            self.route_after_reflection,
            {
                "continue": "think",
                "end": END
            }
        )
        
        return workflow.compile()
        
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke the LangGraph state machine"""
        initial_state = {
            "goal": goal,
            "context": context or {},
            "thoughts": [],
            "plan": [],
            "executed_actions": [],
            "reflection": {},
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "error": None
        }
        
        logger.info(f"🧠 LangGraph started for: {goal}")
        # Run graph to completion
        result_state = await self.graph.ainvoke(initial_state)
        logger.info(f"✅ LangGraph completed: {goal}")
        
        return result_state

    # --- Nodes ---

    async def node_think(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🤔 LangGraph Node: THINK")
        
        context_payload = {
            "goal": state["goal"],
            "context": state["context"],
            "previous_thoughts": state.get("thoughts", []),
            "reflection": state.get("reflection", {})
        }
        
        new_thoughts = await self.llm.generate_thoughts(context_payload)
        
        return {
            "thoughts": new_thoughts
        }

    async def node_plan(self, state: AgentState) -> Dict[str, Any]:
        logger.info("📋 LangGraph Node: PLAN")
        
        context_payload = {
            "goal": state["goal"],
            "thoughts": state.get("thoughts", [])
        }
        
        new_plan = await self.llm.generate_plan(context_payload)
        
        return {
            "plan": new_plan
        }

    async def node_execute(self, state: AgentState) -> Dict[str, Any]:
        logger.info("⚡ LangGraph Node: EXECUTE")
        
        executed_results = []
        plan = state.get("plan", [])
        
        # Execute REAL actions through orchestrator explicitly
        for action in plan:
            action_type = action.get("type", "unknown")
            description = action.get("description", "")
            params = action.get("params", {})
            
            logger.info(f"🔧 Real Execution: {description}")
            
            # Map action types to specific agents in the JARVIS architecture
            target_agent = None
            if action_type in ["research", "scrape", "apply", "job_application", "web_search", "scrape_jobs"]:
                target_agent = AgentType.AUTOMATION
            elif action_type in ["memorize", "store", "retrieve"]:
                target_agent = AgentType.MEMORY
            elif action_type in ["analyze_risk", "permission"]:
                target_agent = AgentType.DECISION
            else:
                target_agent = AgentType.EXECUTION # Default fallback
            
            agent = self.orchestrator.agents.get(target_agent)
            
            result = None
            
            if agent:
                try:
                    # Message the agent across the inner bus
                    message = await self.orchestrator.send_message(
                        to_agent=target_agent,
                        message_type=MessageType.EXECUTE,
                        content={"action": action_type, "params": params},
                        priority=2
                    )
                    
                    # Await response
                    response = await self.orchestrator.message_bus.wait_for_response(
                        correlation_id=message.message_id,
                        timeout=30.0
                    )
                    
                    result = response.content if response else {"error": "Timeout or no response"}
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Agent {target_agent} not available"}
                
            executed_results.append({
                "action": description,
                "type": action_type,
                "params": params,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        return {
            "executed_actions": executed_results
        }

    async def node_reflect(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🪞 LangGraph Node: REFLECT")
        
        context_payload = {
            "goal": state["goal"],
            "actions": state.get("executed_actions", [])[-len(state.get("plan", [])):] # Only reflect on strictly the newest additions
        }
        
        reflection = await self.llm.generate_reflection(context_payload)
        
        return {
            "reflection": reflection,
            "iterations": state["iterations"] + 1
        }

    # --- Edge Routers ---

    def route_after_reflection(self, state: AgentState) -> str:
        reflection = state.get("reflection", {})
        score = reflection.get("score", 0)
        success = reflection.get("success", False)
        
        if success or score > 0.8:
            logger.info("🏁 LangGraph routing to END (Success)")
            return "end"
            
        if state["iterations"] >= state["max_iterations"]:
            logger.info("🏁 LangGraph routing to END (Max Iterations)")
            return "end"
            
        logger.info("🔄 LangGraph routing back to THINK (Iterate)")
        return "continue"
