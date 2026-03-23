import asyncio
import json
import logging
import os
from typing import Dict, Any, AsyncGenerator, Optional

from app.core.message_bus import message_bus
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class BrainController:
    """
    JARVIS V4 Brain Controller - The Central Executive
    Replaces the legacy Orchestrator. Uses async queues, SSE streaming, and parallel tasking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.message_bus = message_bus
        self.task_queue = asyncio.Queue()
        self.agents = {}
        self.background_worker_task = None
        self.cron_task = None
        
        # Initialize fast LLM client for streaming
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            temperature=0.1,
            model_name="llama3-70b-8192", 
            api_key=api_key,
            streaming=True
        ) if api_key else None

    async def initialize(self) -> bool:
        logger.info("🧠 Initializing V4 Brain Controller...")
        self.background_worker_task = asyncio.create_task(self._background_worker())
        self.cron_task = asyncio.create_task(self._cron_scheduler())
        return True

    async def shutdown(self):
        logger.info("🛑 Shutting down Brain Controller...")
        if self.background_worker_task:
            self.background_worker_task.cancel()
        if hasattr(self, 'cron_task') and self.cron_task:
            self.cron_task.cancel()

    async def _cron_scheduler(self):
        """Daily background cron task triggering autonomous digital presence generation"""
        while True:
            try:
                # Wake up every 24 hours (86400 seconds)
                await asyncio.sleep(86400)
                logger.info("⏰ CRON ALARM: Triggering daily Digital Presence Hybrid Pipeline")
                await self.task_queue.put({"intent": "auto_money", "message": "SYSTEM_CRON"})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron error: {e}")
                await asyncio.sleep(60)

    async def process_stream(self, message: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Main V4 entrypoint for handling user input as a stream"""
        try:
            # 1. Fast Intent Routing
            intent = await self._fast_intent_classification(message)
            
            # 2. Trigger Action if actionable
            if intent in ["auto_money", "manage_github", "manage_linkedin", "open_whatsapp", "toggle_wifi"]:
                await self.task_queue.put({"intent": intent, "message": message})
                yield f"data: {json.dumps({'type': 'action', 'content': f'\\n⚡ Trigger: Action [{intent}] queued.\\n'})}\n\n"
                
            # 3. Stream LLM Response
            messages = [
                SystemMessage(content="You are JARVIS V4, an advanced AI. Keep responses extremely concise."),
                HumanMessage(content=message)
            ]
            
            if self.llm:
                async for chunk in self.llm.astream(messages):
                    if chunk.content:
                        payload = {"type": "token", "content": chunk.content}
                        yield f"data: {json.dumps(payload)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': 'LLM Offline (No GROQ_API_KEY)'})}\n\n"

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    async def _fast_intent_classification(self, message: str) -> str:
        """Fast fuzzy intent matching before heavy LLM processing"""
        from app.core.nlp_gateway import nlp_gateway
        
        intent, confidence, _ = nlp_gateway.detect_intent(message)
        if confidence > 0.8:
            return intent
        return "general_chat"

    async def _background_worker(self):
        from app.agents.digital_presence.digital_presence_agent import digital_presence_agent
        
        while True:
            try:
                task = await self.task_queue.get()
                intent = task.get("intent")
                logger.info(f"⚡ Brain Controller Executing background task: {intent}")
                
                # 4. Agent Execution Trigger (If actionable intent)
                if intent in ["auto_money", "manage_github", "manage_linkedin"]:
                    # The original code had:
                    # batch = await digital_presence_agent.prepare_hybrid_batch()
                    # logger.info(f"✅ Digital Presence Batch Ready: {batch['batch_id']} - Waiting for User Appoval.")
                    # In a real system, you'd send a WebSocket alert/notification to the React Native app here.
                    # The provided snippet seems to be for a different method that yields.
                    # For _background_worker, we'll keep the original logic but add the new elifs.
                    batch = await digital_presence_agent.prepare_hybrid_batch()
                    logger.info(f"✅ Digital Presence Batch Ready: {batch['batch_id']} - Waiting for User Appoval.")
                
                # 5. Native OS Execution Trigger (Pillar 7 Mobile Bridge)
                elif intent == "open_whatsapp":
                    from app.core.native_bridge import native_bridge
                    # Note: In production, the NLP Gateway returns exact entity targets
                    fallback_contact = "919876543210" 
                    payload = native_bridge.generate_whatsapp_intent(fallback_contact, "Automated via JARVIS V4 Brain")
                    logger.info(f"📱 Native Bridge: Generated WhatsApp intent for {fallback_contact}")
                    # In a real system, this payload would be sent to the client for native execution.
                    
                elif intent == "toggle_wifi":
                    from app.core.native_bridge import native_bridge
                    payload = native_bridge.generate_wifi_toggle_intent(True)
                    logger.info(f"📱 Native Bridge: Generated WiFi toggle intent")
                    # In a real system, this payload would be sent to the client for native execution.
                
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                await asyncio.sleep(1)
