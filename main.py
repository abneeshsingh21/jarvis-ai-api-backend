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

# Import JARVIS components
from app.core.orchestrator import Orchestrator, SystemState
from app.core.message_bus import message_bus, AgentMessage, MessageType, AgentType
from app.voice.voice_system import create_voice_system_openai, VoiceSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None
voice_system = None


# Pydantic models for API
class UserRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    use_voice: bool = False


class VoiceSessionRequest(BaseModel):
    language: str = "en"


class PermissionResponse(BaseModel):
    request_id: str
    approved: bool
    response_text: Optional[str] = None


class SpecialCommand(BaseModel):
    command: str
    params: Optional[Dict[str, Any]] = None


class ConfigUpdate(BaseModel):
    autonomous_mode: Optional[bool] = None
    user_skills: Optional[list] = None
    auto_apply: Optional[bool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator, voice_system
    
    # Startup
    logger.info("🚀 Starting JARVIS AI Operating System...")
    
    # Load configuration
    config = {
        "agents": {
            "automation": {
                "user_skills": ["Python", "JavaScript", "AI/ML", "Web Development"],
                "auto_apply": False,
                "max_daily_applications": 5
            }
        }
    }
    
    # Initialize orchestrator
    orchestrator = Orchestrator(config=config)
    success = await orchestrator.initialize()
    
    if success:
        logger.info("✅ JARVIS is ready!")
    else:
        logger.error("❌ JARVIS initialization failed")
    
    # Initialize voice system
    # voice_system = create_voice_system_openai(openai_api_key="your-key")
    # await voice_system.initialize()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down JARVIS...")
    if orchestrator:
        await orchestrator.shutdown()
    logger.info("✅ JARVIS shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="JARVIS AI Operating System",
    description="6-Agent AI System with ReAct Reasoning Loop",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "exp://localhost:8081",
        "https://upwork.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if orchestrator and orchestrator.get_state() == SystemState.READY:
        return {
            "status": "healthy",
            "system": "JARVIS",
            "state": orchestrator.get_state().value,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "status": "unhealthy",
            "system": "JARVIS",
            "state": orchestrator.get_state().value if orchestrator else "unknown"
        }


# System info endpoint
@app.get("/system/info")
async def system_info():
    """Get comprehensive system information"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return orchestrator.get_system_info()


# Process user request
@app.post("/api/request")
async def process_request(request: UserRequest):
    """Process a user request through the JARVIS system"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = await orchestrator.process_user_request(
        request=request.message,
        context=request.context,
        use_voice=request.use_voice
    )
    
    return result


# Special commands
@app.post("/api/command")
async def execute_command(command: SpecialCommand):
    """Execute a special system command"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = await orchestrator.handle_special_command(
        command=command.command,
        params=command.params or {}
    )
    
    return result


# Make me money endpoint
@app.post("/api/make-money")
async def make_me_money(background_tasks: BackgroundTasks):
    """Trigger the 'Make me money today' automation"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = await orchestrator.handle_special_command("make_me_money")
    return result


# Toggle autonomous mode
@app.post("/api/autonomous/toggle")
async def toggle_autonomous():
    """Toggle autonomous mode on/off"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = await orchestrator.handle_special_command("toggle_autonomous")
    return result


# Get agent status
@app.get("/api/agents/{agent_type}/status")
async def get_agent_status(agent_type: str):
    """Get status of a specific agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        agent_enum = AgentType(agent_type)
        agent = orchestrator.get_agent(agent_enum)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")
        
        return {
            "agent": agent_type,
            "state": agent.get_state(),
            "metrics": agent.get_metrics()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent type: {agent_type}")


# Get memory info
@app.get("/api/memory")
async def get_memory():
    """Get memory system information"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = await orchestrator.handle_special_command("get_memory")
    return result


# Update configuration
@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update system configuration"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Update automation agent config
    automation_agent = orchestrator.get_agent(AgentType.AUTOMATION)
    if automation_agent:
        if config.user_skills:
            automation_agent.user_skills = config.user_skills
        if config.auto_apply is not None:
            automation_agent.auto_apply_enabled = config.auto_apply
    
    return {"status": "updated"}


# Permission response
@app.post("/api/permission/respond")
async def respond_to_permission(response: PermissionResponse):
    """Respond to a permission request"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    decision_agent = orchestrator.get_agent(AgentType.DECISION)
    if not decision_agent:
        raise HTTPException(status_code=503, detail="Decision agent not available")
    
    success = await decision_agent.respond_to_permission(
        request_id=response.request_id,
        approved=response.approved,
        user_response=response.response_text
    )
    
    return {"success": success}


# Get pending permissions
@app.get("/api/permissions/pending")
async def get_pending_permissions():
    """Get all pending permission requests"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    decision_agent = orchestrator.get_agent(AgentType.DECISION)
    if not decision_agent:
        return {"permissions": []}
    
    return {"permissions": decision_agent.get_pending_permissions()}


# Job discovery endpoint
@app.post("/api/jobs/discover")
async def discover_jobs(
    platforms: list = None,
    keywords: list = None,
    limit: int = 10
):
    """Discover jobs from freelancing platforms"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    automation_agent = orchestrator.get_agent(AgentType.AUTOMATION)
    if not automation_agent:
        raise HTTPException(status_code=503, detail="Automation agent not available")
    
    jobs = await automation_agent.discover_jobs(
        platforms=platforms,
        keywords=keywords,
        limit=limit
    )
    
    return {"jobs": [j.to_dict() for j in jobs]}


# Generate proposal endpoint
@app.post("/api/jobs/{job_id}/proposal")
async def generate_proposal(job_id: str, custom_info: Optional[Dict] = None):
    """Generate a proposal for a job"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    automation_agent = orchestrator.get_agent(AgentType.AUTOMATION)
    if not automation_agent:
        raise HTTPException(status_code=503, detail="Automation agent not available")
    
    proposal = await automation_agent.generate_proposal(job_id, custom_info)
    
    if not proposal:
        raise HTTPException(status_code=404, detail="Job not found or proposal generation failed")
    
    return {"proposal": proposal.to_dict()}


# Apply to job endpoint
@app.post("/api/jobs/{job_id}/apply")
async def apply_to_job(job_id: str, proposal_id: Optional[str] = None):
    """Apply to a job"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    automation_agent = orchestrator.get_agent(AgentType.AUTOMATION)
    if not automation_agent:
        raise HTTPException(status_code=503, detail="Automation agent not available")
    
    result = await automation_agent.apply_to_job(job_id, proposal_id)
    
    return {"applied": result}


# WebSocket for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message_type = data.get("type", "message")
            
            if message_type == "message":
                # Process user message
                result = await orchestrator.process_user_request(
                    request=data.get("message", ""),
                    context=data.get("context"),
                    use_voice=data.get("use_voice", False)
                )
                
                await websocket.send_json({
                    "type": "response",
                    "data": result
                })
            
            elif message_type == "voice_start":
                # Start voice session
                result = await orchestrator.handle_special_command(
                    "start_voice_session",
                    {"language": data.get("language", "en")}
                )
                await websocket.send_json({
                    "type": "voice_session_started",
                    "data": result
                })
            
            elif message_type == "voice_data":
                # Process voice data
                # This would handle streaming audio
                pass
            
            elif message_type == "interrupt":
                # Interrupt current speech
                if voice_system:
                    await voice_system.interrupt()
                await websocket.send_json({
                    "type": "interrupted"
                })
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Voice WebSocket for streaming audio
@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice streaming"""
    await websocket.accept()
    
    session: Optional[VoiceSession] = None
    
    try:
        while True:
            message = await websocket.receive()
            
            if isinstance(message, bytes):
                # Audio data
                if session:
                    transcript = await session.process_audio(message)
                    if transcript:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript
                        })
                        
                        # Process through orchestrator
                        result = await orchestrator.process_user_request(
                            request=transcript,
                            use_voice=True
                        )
                        
                        # Speak response
                        if result.get("success"):
                            response_text = result.get("response_text", "Done")
                            await session.speak(response_text)
            
            else:
                # JSON message
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "start_session":
                    language = data.get("language", "en")
                    session = await voice_system.create_session(language)
                    
                    # Set up callbacks
                    session.on_transcript = lambda text, role: websocket.send_json({
                        "type": "transcript",
                        "text": text,
                        "role": role
                    })
                    
                    session.on_audio_output = lambda audio: websocket.send_bytes(audio)
                    
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session.session_id
                    })
                
                elif msg_type == "end_session":
                    if session:
                        await voice_system.end_session(session.session_id)
                        session = None
                    await websocket.send_json({"type": "session_ended"})
                
                elif msg_type == "interrupt":
                    if session:
                        await session.interrupt()
    
    except WebSocketDisconnect:
        if session:
            await voice_system.end_session(session.session_id)
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
