"""
JARVIS Memory Agent
Manages long-term and short-term memory, context retrieval, learning
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np

from app.core.base_agent import BaseAgent, AgentState
from app.core.message_bus import (
    AgentMessage, MessageType, AgentType, MessageBuilder, message_bus
)
from .db_manager import MemoryDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry"""
    memory_id: str
    content: Any
    memory_type: str  # "fact", "experience", "preference", "context", "learned"
    importance: int  # 1-10
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: str = None
    last_accessed: str = None
    access_count: int = 0
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = set()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "tags": list(self.tags)
        }


class VectorStore:
    """Simple in-memory vector store for semantic search"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add(self, memory_id: str, embedding: List[float], metadata: Dict = None):
        """Add a vector to the store"""
        self.vectors[memory_id] = np.array(embedding)
        self.metadata[memory_id] = metadata or {}
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[tuple]:
        """Search for similar vectors using cosine similarity"""
        if not self.vectors:
            return []
        
        query = np.array(query_embedding)
        
        # Calculate cosine similarity
        similarities = []
        for memory_id, vector in self.vectors.items():
            similarity = self._cosine_similarity(query, vector)
            if similarity >= threshold:
                similarities.append((memory_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def delete(self, memory_id: str):
        """Delete a vector from the store"""
        if memory_id in self.vectors:
            del self.vectors[memory_id]
            del self.metadata[memory_id]


class MemoryAgent(BaseAgent):
    """
    Memory Agent: Manages all memory operations
    
    Capabilities:
    - Store and retrieve memories
    - Semantic search using embeddings
    - Context-aware memory retrieval
    - Memory consolidation and cleanup
    - Learning from experiences
    """
    
    def __init__(self, embedding_client=None, config: Dict = None):
        super().__init__(AgentType.MEMORY, config=config)
        self.embedding_client = embedding_client
        
        # Memory stores
        self.short_term: Dict[str, MemoryEntry] = {}  # Session memories
        self.long_term: Dict[str, MemoryEntry] = {}   # Persistent memories
        self.working_memory: Dict[str, Any] = {}      # Current context
        
        # Vector store for semantic search
        self.vector_store = VectorStore()
        
        # Memory management settings
        self.short_term_limit = config.get("short_term_limit", 100) if config else 100
        self.long_term_limit = config.get("long_term_limit", 10000) if config else 10000
        self.decay_hours = config.get("decay_hours", 24) if config else 24
        
        # Conversation context
        self.conversation_history: List[Dict] = []
        self.max_history = 50
        
        self.db = MemoryDB(config.get("db_path", "jarvis_memory.db") if config else "jarvis_memory.db")
        # User profile loaded from DB later
        self.user_profile: Dict[str, Any] = {
            "preferences": {},
            "habits": {},
            "facts": {},
            "goals": []
        }
    
    async def _initialize(self) -> bool:
        """Initialize the Memory Agent"""
        logger.info("🧠 Memory Agent initializing...")
        
        # Register message handlers
        self.register_handler(MessageType.MEMORY_STORE, self._handle_memory_store)
        self.register_handler(MessageType.MEMORY_RETRIEVE, self._handle_memory_retrieve)
        self.register_handler(MessageType.MEMORY_UPDATE, self._handle_memory_update)
        
        # Load from DB
        memories = self.db.load_all_memories()
        
        # Hydrate short term
        for m in memories["short_term"]:
            entry = MemoryEntry(**m)
            self.short_term[entry.memory_id] = entry
            if entry.embeddings:
                self.vector_store.add(entry.memory_id, entry.embeddings, entry.to_dict())
                
        # Hydrate long term
        for m in memories["long_term"]:
            entry = MemoryEntry(**m)
            self.long_term[entry.memory_id] = entry
            if entry.embeddings:
                self.vector_store.add(entry.memory_id, entry.embeddings, entry.to_dict())
                
        # Load profile
        self.user_profile = self.db.load_profile()
        logger.info(f"💾 Loaded {len(self.short_term)} short-term and {len(self.long_term)} long-term memories from DB")

        # Start maintenance loop
        asyncio.create_task(self._maintenance_loop())
        
        logger.info("✅ Memory Agent initialized")
        return True
    
    async def _cleanup(self):
        """Cleanup resources"""
        # Save memories before shutdown
        await self._consolidate_memories()
    
    async def _handle_message(self, message: AgentMessage):
        """Handle generic messages"""
        logger.debug(f"📥 Memory Agent received: {message.message_type}")
    
    async def _handle_memory_store(self, message: AgentMessage):
        """Handle memory store requests"""
        content = message.content
        key = content.get("key", "")
        data = content.get("data")
        importance = content.get("importance", 5)
        memory_type = content.get("memory_type", "fact")
        metadata = content.get("metadata", {})
        
        logger.info(f"💾 Memory store: {key}")
        
        # Store the memory
        memory_id = await self.store(
            content=data,
            memory_type=memory_type,
            importance=importance,
            metadata={**metadata, "key": key, "source": message.from_agent}
        )
        
        # Send confirmation
        await self.send_response(
            original_message=message,
            content={"memory_id": memory_id, "stored": True}
        )
    
    async def _handle_memory_retrieve(self, message: AgentMessage):
        """Handle memory retrieve requests"""
        content = message.content
        query = content.get("query", "")
        context = content.get("context", {})
        limit = content.get("limit", 5)
        
        logger.info(f"🔍 Memory retrieve: {query}")
        
        # Retrieve memories
        memories = await self.retrieve(
            query=query,
            context=context,
            limit=limit
        )
        
        # Send response
        await self.send_response(
            original_message=message,
            content={"memories": [m.to_dict() for m in memories]}
        )
    
    async def _handle_memory_update(self, message: AgentMessage):
        """Handle memory update requests"""
        content = message.content
        memory_id = content.get("memory_id", "")
        updates = content.get("updates", {})
        
        logger.info(f"📝 Memory update: {memory_id}")
        
        success = await self.update(memory_id, updates)
        
        await self.send_response(
            original_message=message,
            content={"updated": success}
        )
    
    async def store(
        self,
        content: Any,
        memory_type: str = "fact",
        importance: int = 5,
        metadata: Dict = None,
        tags: List[str] = None
    ) -> str:
        """Store a new memory"""
        # Generate memory ID
        content_hash = hashlib.md5(
            json.dumps(content, default=str).encode()
        ).hexdigest()[:12]
        memory_id = f"mem_{memory_type}_{content_hash}_{datetime.utcnow().timestamp()}"
        
        # Generate embeddings if client available
        embeddings = None
        if self.embedding_client:
            try:
                text_content = json.dumps(content, default=str)
                embeddings = await self.embedding_client.embed(text_content)
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")
        
        # Create memory entry
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embeddings=embeddings,
            metadata=metadata or {},
            tags=set(tags or [])
        )
        
        # Store in appropriate location
        if importance >= 7 or memory_type in ["preference", "goal"]:
            self.long_term[memory_id] = entry
            self.db.save_memory(entry.to_dict(), is_long_term=True)
            
            # Add to vector store if embeddings available
            if embeddings:
                self.vector_store.add(memory_id, embeddings, entry.to_dict())
        else:
            self.short_term[memory_id] = entry
            self.db.save_memory(entry.to_dict(), is_long_term=False)
        
        # Update user profile for certain memory types
        if memory_type == "preference":
            self._update_user_profile(content, metadata)
        
        logger.info(f"💾 Stored memory: {memory_id}")
        return memory_id
    
    async def retrieve(
        self,
        query: str,
        context: Dict = None,
        limit: int = 5,
        memory_type: str = None,
        min_importance: int = 1
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        results = []
        
        # 1. Semantic search if embeddings available
        if self.embedding_client:
            try:
                query_embedding = await self.embedding_client.embed(query)
                similar = self.vector_store.search(query_embedding, top_k=limit)
                
                for memory_id, similarity in similar:
                    if memory_id in self.long_term:
                        entry = self.long_term[memory_id]
                        entry.access_count += 1
                        entry.last_accessed = datetime.utcnow().isoformat()
                        results.append(entry)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # 2. Keyword search in short-term memory
        query_lower = query.lower()
        for entry in self.short_term.values():
            content_str = json.dumps(entry.content, default=str).lower()
            if query_lower in content_str:
                if entry.importance >= min_importance:
                    if memory_type is None or entry.memory_type == memory_type:
                        entry.access_count += 1
                        entry.last_accessed = datetime.utcnow().isoformat()
                        if entry not in results:
                            results.append(entry)
        
        # 3. Context-based retrieval
        if context:
            context_results = await self._retrieve_by_context(context, limit)
            for entry in context_results:
                if entry not in results:
                    results.append(entry)
        
        # Sort by importance and recency
        results.sort(key=lambda e: (e.importance, e.last_accessed), reverse=True)
        
        return results[:limit]
    
    async def _retrieve_by_context(
        self,
        context: Dict,
        limit: int
    ) -> List[MemoryEntry]:
        """Retrieve memories based on context"""
        results = []
        
        # Check for relevant tags
        context_tags = set(context.get("tags", []))
        
        for entry in list(self.short_term.values()) + list(self.long_term.values()):
            if entry.tags & context_tags:  # Intersection
                results.append(entry)
        
        # Check for related topics
        topic = context.get("topic")
        if topic:
            for entry in list(self.long_term.values()):
                if entry.metadata.get("topic") == topic:
                    if entry not in results:
                        results.append(entry)
        
        return results[:limit]
    
    async def update(self, memory_id: str, updates: Dict) -> bool:
        """Update an existing memory"""
        entry = self.short_term.get(memory_id) or self.long_term.get(memory_id)
        
        if not entry:
            return False
        
        if "content" in updates:
            entry.content = updates["content"]
        if "importance" in updates:
            entry.importance = updates["importance"]
        if "metadata" in updates:
            entry.metadata.update(updates["metadata"])
        if "tags" in updates:
            entry.tags.update(updates["tags"])
        
        entry.last_accessed = datetime.utcnow().isoformat()
        self.db.save_memory(entry.to_dict(), is_long_term=(memory_id in self.long_term))
        
        logger.info(f"📝 Updated memory: {memory_id}")
        return True
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id in self.short_term:
            del self.short_term[memory_id]
            self.db.delete_memory(memory_id)
            logger.info(f"🗑️ Deleted short-term memory: {memory_id}")
            return True
        
        if memory_id in self.long_term:
            del self.long_term[memory_id]
            self.db.delete_memory(memory_id)
            self.vector_store.delete(memory_id)
            logger.info(f"🗑️ Deleted long-term memory: {memory_id}")
            return True
        
        return False
    
    def add_to_conversation_history(
        self,
        role: str,
        content: str,
        metadata: Dict = None
    ):
        """Add entry to conversation history"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(entry)
        
        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation context"""
        return self.conversation_history[-limit:]
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        return self.user_profile
    
    def update_user_profile(self, updates: Dict[str, Any]):
        """Update user profile"""
        for key, value in updates.items():
            if key in self.user_profile:
                if isinstance(self.user_profile[key], dict):
                    self.user_profile[key].update(value)
                elif isinstance(self.user_profile[key], list):
                    self.user_profile[key].extend(value)
                else:
                    self.user_profile[key] = value
        self.db.save_profile(self.user_profile)
    
    def _update_user_profile(self, content: Any, metadata: Dict):
        """Update user profile from memory content"""
        pref_type = metadata.get("preference_type")
        
        if pref_type:
            self.user_profile["preferences"][pref_type] = content
            self.db.save_profile(self.user_profile)
    
    async def _maintenance_loop(self):
        """Periodic memory maintenance"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._consolidate_memories()
                await self._cleanup_expired_memories()
            except Exception as e:
                logger.error(f"❌ Memory maintenance error: {e}")
    
    async def _consolidate_memories(self):
        """Consolidate short-term memories to long-term"""
        logger.info("🔄 Consolidating memories...")
        
        # Move high-importance short-term memories to long-term
        to_move = []
        for memory_id, entry in self.short_term.items():
            if entry.importance >= 7 or entry.access_count >= 3:
                to_move.append(memory_id)
        
        for memory_id in to_move:
            entry = self.short_term.pop(memory_id)
            self.long_term[memory_id] = entry
            self.db.save_memory(entry.to_dict(), is_long_term=True)
            
            if entry.embeddings:
                self.vector_store.add(memory_id, entry.embeddings, entry.to_dict())
        
        logger.info(f"🔄 Consolidated {len(to_move)} memories")
    
    async def _cleanup_expired_memories(self):
        """Clean up expired short-term memories"""
        logger.info("🧹 Cleaning up expired memories...")
        
        cutoff = datetime.utcnow() - timedelta(hours=self.decay_hours)
        to_delete = []
        
        for memory_id, entry in self.short_term.items():
            created = datetime.fromisoformat(entry.created_at)
            if created < cutoff and entry.access_count == 0:
                to_delete.append(memory_id)
        
        for memory_id in to_delete:
            del self.short_term[memory_id]
            self.db.delete_memory(memory_id)
        
        logger.info(f"🧹 Cleaned up {len(to_delete)} expired memories")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "vector_store_count": len(self.vector_store.vectors),
            "conversation_history_count": len(self.conversation_history),
            "user_profile_keys": list(self.user_profile.keys())
        }
    
    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Search memories by tag"""
        results = []
        
        for entry in list(self.short_term.values()) + list(self.long_term.values()):
            if tag in entry.tags:
                results.append(entry)
        
        return results
    
    async def learn_from_interaction(
        self,
        user_input: str,
        agent_response: str,
        outcome: str
    ):
        """Learn from user interactions"""
        # Store the interaction
        await self.store(
            content={
                "user_input": user_input,
                "agent_response": agent_response,
                "outcome": outcome
            },
            memory_type="experience",
            importance=6,
            metadata={"interaction_type": "dialogue"},
            tags=["learning", "interaction"]
        )
        
        # Extract preferences if applicable
        if "prefer" in user_input.lower() or "like" in user_input.lower():
            await self.store(
                content=user_input,
                memory_type="preference",
                importance=8,
                metadata={"preference_type": "user_stated"}
            )
