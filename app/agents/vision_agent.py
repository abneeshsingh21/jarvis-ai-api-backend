import os
import logging
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class VisionAgent:
    """
    JARVIS V4 - Multi-Modal Vision Agent
    Analyzes camera frames or uploaded images via GPT-4o Vision to provide real-time scene understanding.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=300,
            api_key=api_key
        ) if api_key else None

    async def analyze_scene(self, base64_image: str, query: str = "Describe what I am looking at.") -> str:
        """
        Takes a base64 encoded image from the mobile client and a text prompt,
        returning the LLM's visual analysis.
        """
        logger.info("👁️ Vision Agent analyzing new camera frame string...")
        
        if not self.llm:
            logger.warning("OPENAI_API_KEY missing. Cannot perform vision inference.")
            return "Vision system offline: Missing API Key."

        # Format image for GPT-4 Vision payload
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"You are JARVIS V4 Vision Module. {query}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
        
        try:
            response = await self.llm.ainvoke([message])
            logger.info("✅ Vision inference complete.")
            return response.content
        except Exception as e:
            logger.error(f"Vision Agent error: {e}")
            return f"Error analyzing image: {str(e)}"

# Global Singleton
vision_agent = VisionAgent()
