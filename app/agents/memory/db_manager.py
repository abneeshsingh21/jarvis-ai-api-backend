import sqlite3
import json
import logging
from typing import Dict, Any, List
import os

logger = logging.getLogger(__name__)

class MemoryDB:
    """SQLite wrapper for persisting JARVIS MemoryAgent state permanently."""
    
    def __init__(self, db_path: str = "jarvis_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        content TEXT,
                        memory_type TEXT,
                        importance INTEGER,
                        embeddings TEXT,
                        metadata TEXT,
                        created_at TEXT,
                        last_accessed TEXT,
                        access_count INTEGER,
                        tags TEXT,
                        is_long_term BOOLEAN
                    )
                ''')
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS profiles (
                        key TEXT PRIMARY KEY,
                        data TEXT
                    )
                ''')
                conn.commit()
                logger.info(f"📂 MemoryDB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MemoryDB: {e}")

    def save_memory(self, entry: Dict[str, Any], is_long_term: bool = False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO memories 
                    (memory_id, content, memory_type, importance, embeddings, metadata, created_at, last_accessed, access_count, tags, is_long_term)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry["memory_id"],
                    json.dumps(entry["content"], default=str),
                    entry["memory_type"],
                    entry["importance"],
                    json.dumps(entry.get("embeddings")) if entry.get("embeddings") is not None else None,
                    json.dumps(entry.get("metadata", {})),
                    entry.get("created_at"),
                    entry.get("last_accessed"),
                    entry.get("access_count", 0),
                    json.dumps(list(entry.get("tags", []))),
                    is_long_term
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save memory to DB: {e}")

    def delete_memory(self, memory_id: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM memories WHERE memory_id = ?', (memory_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to delete memory from DB: {e}")

    def load_all_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        memories = {"short_term": [], "long_term": []}
        try:
            if not os.path.exists(self.db_path):
                return memories
                
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM memories')
                for row in cursor:
                    entry = dict(row)
                    entry["content"] = json.loads(entry["content"])
                    entry["embeddings"] = json.loads(entry["embeddings"]) if entry["embeddings"] else None
                    entry["metadata"] = json.loads(entry["metadata"])
                    entry["tags"] = set(json.loads(entry["tags"]))
                    
                    if entry.pop("is_long_term"):
                        memories["long_term"].append(entry)
                    else:
                        memories["short_term"].append(entry)
        except Exception as e:
            logger.error(f"❌ Failed to load memories from DB: {e}")
            
        return memories
        
    def save_profile(self, profile: Dict[str, Any]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('INSERT OR REPLACE INTO profiles (key, data) VALUES (?, ?)', 
                           ("user_profile", json.dumps(profile)))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save profile to DB: {e}")
            
    def load_profile(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT data FROM profiles WHERE key = ?', ("user_profile",))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.error(f"❌ Failed to load profile from DB: {e}")
            
        return {
            "preferences": {},
            "habits": {},
            "facts": {},
            "goals": []
        }
