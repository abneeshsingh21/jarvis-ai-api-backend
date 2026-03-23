import os
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class PremiumTTSEngine:
    """
    JARVIS V4 Premium TTS Layer
    Designed to stream audio using ElevenLabs or OpenAI TTS for natural human-like responses.
    Includes background environmental noise handler stubs.
    """
    def __init__(self):
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        # Rachel voice ID from ElevenLabs is popular for AI
        self.voice_id = "21m00Tcm4TlvDq8ikWAM" 

    def get_wake_word_client_keys(self) -> dict:
        """
        Porcupine Wake Word is actually executed on the Mobile Client 
        to prevent 24/7 battery drain and network streaming.
        This provides the access keys for the client to initialize securely.
        """
        return {
            "picovoice_access_key": os.getenv("PICOVOICE_ACCESS_KEY", "pending_key")
        }
        
    async def stream_tts(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Takes an LLM async text stream and yields audio chunks for the mobile app real-time playing.
        """
        if not self.elevenlabs_key:
            logger.warning("No ElevenLabs Key available. Yielding mock TTS audio chunks.")
        
        # Buffer to hold text chunks until a complete sentence/phrase is formed
        # before sending to TTS API for proper prosody.
        text_buffer = ""
        
        async for text_chunk in text_stream:
            text_buffer += text_chunk
            # Simple heuristic: send to TTS API when sentence ends
            if any(char in text_chunk for char in ['.', '!', '?', '\n']):
                # Mock sending to ElevenLabs WebSocket API
                # In production: send text_buffer to API, yield audio chunk
                yield f"mock_audio_chunk:{text_buffer}".encode("utf-8")
                text_buffer = ""

# Global Singleton
tts_engine = PremiumTTSEngine()
