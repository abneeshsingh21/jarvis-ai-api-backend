import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class HinglishNLPGateway:
    """
    JARVIS V4 - Fast Hinglish NLP Gateway
    Bypasses LLM latency by intercepting core intents via local pattern matching.
    Supports English & Hinglish (Hindi + English).
    """
    
    def __init__(self):
        # Maps regular expressions to Intent IDs and extraction groups
        self.intent_patterns = {
            # Automation & Earning
            "auto_money": [
                r"\b(money|paisa|kamana|job|upwork|fiverr|freelance)\b",
                r"paisa kaise kamau",
                r"make me money"
            ],
            # Social Media
            "manage_linkedin": [
                r"\b(linkedin|post|brand|reach)\b",
                r"post on linkedin",
                r"linkedin par post dalo"
            ],
            # Developer Tasks
            "manage_github": [
                r"\b(github|repo|commit|code|push|pull)\b",
                r"code push kardo"
            ],
            # Mobile Access & OS 
            "read_calendar": [
                r"\b(schedule|calendar|meeting|appointment|remind)\b",
                r"aaj ka schedule kya hai"
            ],
            "toggle_wifi": [
                r"turn (on|off) wifi",
                r"wifi (chalu|band) karo"
            ],
            "open_whatsapp": [
                r"\b(whatsapp|message|send)\b.*to (?P<contact>\w+)",
                r"(?P<contact>\w+) ko message bhejo"
            ]
        }

    def detect_intent(self, user_input: str) -> Tuple[str, float, Dict[str, str]]:
        """
        Detects intent based on regex patterns.
        Returns: (intent_name, confidence_score, extracted_entities)
        """
        user_input_lower = user_input.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    logger.info(f"⚡ Fast NLP Match: {intent}")
                    return intent, 0.95, match.groupdict()
                    
        return "general_chat", 0.0, {}

# Global Singleton
nlp_gateway = HinglishNLPGateway()
