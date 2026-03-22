import os
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# --- ReAct Loop Schemas ---
class ThoughtSchema(BaseModel):
    content: str = Field(description="The actual thought or analysis of the goal.")
    reasoning_type: str = Field(description="Type: 'analysis', 'hypothesis', 'conclusion', or 'question'")
    confidence: float = Field(description="Between 0.0 and 1.0 indicating confidence in this thought.")

class NodeActionSchema(BaseModel):
    type: str = Field(description="The action type, e.g., 'research', 'scrape_jobs', 'execute', 'web_search'")
    description: str = Field(description="Human readable description of what this action does.")
    params: Dict[str, Any] = Field(description="Key-value parameters to pass to the action.")

class ThoughtsList(BaseModel):
    thoughts: List[ThoughtSchema]

class PlanSchema(BaseModel):
    actions: List[NodeActionSchema]

class ReflectionSchema(BaseModel):
    success: bool = Field(description="Did the execution succeed in achieving the goal?")
    observations: List[str] = Field(description="Factual observations of what happened.")
    lessons_learned: List[str] = Field(description="What should be done differently next time?")
    improvements: List[str] = Field(description="Concrete steps to improve the execution.")
    score: float = Field(description="Success score from 0.0 to 1.0.")

# --- Generic LLM Interface ---
class LLMClient:
    """
    Production LLM Client wrapping LangChain Groq
    Hooks straight into Groq for 800+ Tokens/Sec structured JSON outputs.
    """
    def __init__(self, api_key: str = None, model: str = 'llama-3.1-8b-instant'):
        os.environ['GROQ_API_KEY'] = api_key or os.getenv("GROQ_API_KEY", "")
        
        self.llm = ChatGroq(model=model, temperature=0.2)
        
        self.think_agent = self.llm.with_structured_output(ThoughtsList)
        self.plan_agent = self.llm.with_structured_output(PlanSchema)
        self.reflect_agent = self.llm.with_structured_output(ReflectionSchema)

    async def generate_thoughts(self, context: Dict) -> List[Dict]:
        try:
            prompt = f"Analyze the following goal and context. Break down the reasoning into a logical chain.\nGoal: {context.get('goal')}\nContext: {context}"
            result = await self.think_agent.ainvoke(prompt)
            # Serialize
            return [
                {
                    "content": t.content,
                    "type": t.reasoning_type,
                    "confidence": t.confidence
                } for t in result.thoughts
            ]
        except Exception as e:
            logger.error(f"LangChain Think Error: {e}")
            return [{"content": f"Failed to think: {e}", "type": "error", "confidence": 0.0}]

    async def generate_plan(self, context: Dict) -> List[Dict]:
        try:
            prompt = f"Create an actionable, strict JSON plan containing actions to achieve the goal.\nGoal: {context.get('goal')}\nThoughts: {context.get('thoughts')}"
            result = await self.plan_agent.ainvoke(prompt)
            return [
                {
                    "type": a.type,
                    "description": a.description,
                    "params": a.params
                } for a in result.actions
            ]
        except Exception as e:
            logger.error(f"LangChain Plan Error: {e}")
            return []

    async def generate_reflection(self, context: Dict) -> Dict:
        try:
            prompt = f"Reflect on the executed actions vs the intended goal.\nGoal: {context.get('goal')}\nExecution Log: {context.get('actions')}"
            result = await self.reflect_agent.ainvoke(prompt)
            return {
                "success": result.success,
                "observations": result.observations,
                "lessons_learned": result.lessons_learned,
                "improvements": result.improvements,
                "score": result.score
            }
        except Exception as e:
            logger.error(f"LangChain Reflect Error: {e}")
            return {"success": False, "observations": [], "lessons_learned": [], "improvements": [], "score": 0.0}
