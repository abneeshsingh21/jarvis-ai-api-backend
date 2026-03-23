"""
JARVIS AI Operating System - Main FastAPI Application
Production-ready API for the 6-Agent AI System
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import JARVIS components (V4)
from app.core.brain_controller import BrainController

from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global controller instance
brain_controller: Optional[BrainController] = None

# Pydantic models for API
class UserRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    use_voice: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global brain_controller
    
    # Startup
    logger.info("🚀 Starting JARVIS V4 Brain Controller...")
    
    # Initialize V4 brain controller
    brain_controller = BrainController()
    success = await brain_controller.initialize()
    
    if success:
        logger.info("✅ JARVIS V4 is ready!")
    else:
        logger.error("❌ JARVIS V4 initialization failed")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down JARVIS...")
    if brain_controller:
        await brain_controller.shutdown()
    logger.info("✅ JARVIS shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="JARVIS AI Operating System V4",
    description="Real-Time Autonomous Brain Controller System",
    version="4.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "JARVIS V4",
        "timestamp": datetime.utcnow().isoformat()
    }

# V4 Real-Time Streaming Endpoint
@app.post("/api/v4/stream")
async def process_stream_v4(request: UserRequest):
    """Process user request and stream LLM tokens directly to frontend via SSE"""
    if not brain_controller:
        raise HTTPException(status_code=503, detail="V4 Brain Controller not initialized")
    
    return StreamingResponse(
        brain_controller.process_stream(request.message, request.context),
        media_type="text/event-stream"
    )

# Legacy Endpoints disabled for V4 Migration
@app.post("/api/request")
async def process_request(request: UserRequest):
    raise HTTPException(status_code=400, detail="V3 endpoint deprecated. Use /api/v4/stream")

# V4 Multi-Modal Vision Endpoint
class VisionRequest(BaseModel):
    image_base64: str
    query: Optional[str] = "Describe what I am looking at."

@app.post("/api/v4/vision/analyze")
async def analyze_vision_frame(request: VisionRequest):
    """Receive a base64 encoded image frame from React Native and process it via GPT-4o Vision"""
    from app.agents.vision_agent import vision_agent
    result = await vision_agent.analyze_scene(request.image_base64, request.query)
    return {"status": "success", "analysis": result}


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
