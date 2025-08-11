import hashlib
import time
import json
import redis
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import logging

@dataclass
class MemoryNode:
    """Represents a single memory entry with temporal and consensus metadata"""
    snowflake_id: str
    content: str
    confidence_score: float
    creation_timestamp: int
    last_accessed: int
    access_count: int
    consensus_weight: float
    contradiction_flags: List[str]
    source_context: str
    
class SnowflakeGenerator:
    """Twitter-style snowflake ID generator for memory entries"""
    
    def __init__(self, worker_id: int = 1, datacenter_id: int = 1):
        self.worker_id = worker_id & 0x1F  # 5 bits
        self.datacenter_id = datacenter_id & 0x1F  # 5 bits
        self.sequence = 0
        self.last_timestamp = -1
        self.epoch = 1288834974657  # Twitter epoch
    
    def generate(self) -> str:
        """Generate a unique snowflake ID"""
        timestamp = int(time.time() * 1000)
        
        if timestamp < self.last_timestamp:
            raise Exception("Clock moved backwards")
        
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 0xFFF  # 12 bits
            if self.sequence == 0:
                timestamp = self._wait_next_millis(timestamp)
        else:
            self.sequence = 0
        
        self.last_timestamp = timestamp
        
        # Build the ID: 41 bits timestamp + 5 bits datacenter + 5 bits worker + 12 bits sequence
        snowflake = ((timestamp - self.epoch) << 22) | (self.datacenter_id << 17) | (self.worker_id << 12) | self.sequence
        return hex(snowflake)
    
    def _wait_next_millis(self, last_timestamp: int) -> int:
        timestamp = int(time.time() * 1000)
        while timestamp <= last_timestamp:
            timestamp = int(time.time() * 1000)
        return timestamp

class ConsensusGraph:
    """Manages consensus knowledge graph for memory validation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.consensus_threshold = 0.7
        
    async def update_consensus(self, memory_id: str, related_memories: List[str]) -> float:
        """Update consensus score based on related memory agreement"""
        consensus_scores = []
        
        for related_id in related_memories:
            related_memory = await self._get_memory_by_id(related_id)
            if related_memory:
                # Calculate semantic similarity (simplified with content hash comparison)
                similarity = self._calculate_similarity(memory_id, related_id)
                consensus_scores.append(similarity * related_memory.confidence_score)
        
        if not consensus_scores:
            return 0.5  # Neutral consensus for isolated memories
            
        return np.mean(consensus_scores)
    
    def _calculate_similarity(self, memory_id1: str, memory_id2: str) -> float:
        """Simplified similarity calculation - in production use vector embeddings"""
        # This is a placeholder - implement proper semantic similarity
        hash1 = hashlib.md5(memory_id1.encode()).hexdigest()
        hash2 = hashlib.md5(memory_id2.encode()).hexdigest()
        
        # Simple Hamming distance for demonstration
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return max(0, 1 - (distance / len(hash1)))
    
    async def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryNode]:
        """Fetch memory node from storage"""
        memory_data = self.redis.get(f"memory:{memory_id}")
        if memory_data:
            return MemoryNode(**json.loads(memory_data))
        return None

class ReconsiderationEngine:
    """Core engine for memory reconsideration and evolution"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.snowflake_gen = SnowflakeGenerator()
        self.consensus_graph = ConsensusGraph(redis_client)
        self.logger = logging.getLogger(__name__)
        
        # Reconsideration parameters
        self.temporal_decay_factor = 0.95
        self.reconsideration_threshold = 0.3
        self.max_reconsideration_depth = 5
    
    async def store_memory(self, content: str, source_context: str, initial_confidence: float = 0.8) -> str:
        """Store a new memory with snowflake ID"""
        snowflake_id = self.snowflake_gen.generate()
        current_time = int(time.time())
        
        memory = MemoryNode(
            snowflake_id=snowflake_id,
            content=content,
            confidence_score=initial_confidence,
            creation_timestamp=current_time,
            last_accessed=current_time,
            access_count=1,
            consensus_weight=0.5,  # Neutral until consensus is established
            contradiction_flags=[],
            source_context=source_context
        )
        
        # Store in Redis
        self.redis.setex(
            f"memory:{snowflake_id}",
            timedelta(days=365),  # TTL for memory persistence
            json.dumps(asdict(memory))
        )
        
        # Index for reconsideration queue
        self.redis.zadd("reconsideration_queue", {snowflake_id: current_time})
        
        self.logger.info(f"Stored memory {snowflake_id} with confidence {initial_confidence}")
        return snowflake_id
    
    async def reconsider_memory(self, memory_id: str) -> Tuple[bool, float]:
        """
        Core reconsideration algorithm:
        1. Retrieve memory
        2. Apply temporal decay
        3. Check consensus against knowledge graph
        4. Update or flag for revision
        """
        memory = await self._get_memory(memory_id)
        if not memory:
            return False, 0.0
        
        # Apply temporal decay
        time_elapsed = int(time.time()) - memory.creation_timestamp
        days_elapsed = time_elapsed / (24 * 3600)
        temporal_confidence = memory.confidence_score * (self.temporal_decay_factor ** days_elapsed)
        
        # Get related memories for consensus check
        related_memories = await self._find_related_memories(memory_id, memory.content)
        consensus_score = await self.consensus_graph.update_consensus(memory_id, related_memories)
        
        # Detect contradictions
        contradictions = await self._detect_contradictions(memory, related_memories)
        
        # Calculate new confidence
        new_confidence = self._calculate_reconsidered_confidence(
            temporal_confidence, consensus_score, len(contradictions)
        )
        
        # Update memory
        memory.confidence_score = new_confidence
        memory.consensus_weight = consensus_score
        memory.contradiction_flags = contradictions
        memory.last_accessed = int(time.time())
        memory.access_count += 1
        
        # Decide if memory needs updating
        needs_update = new_confidence < self.reconsideration_threshold
        
        if needs_update:
            self.logger.warning(f"Memory {memory_id} flagged for update: confidence={new_confidence:.3f}")
            self.redis.sadd("memories_to_update", memory_id)
        
        # Store updated memory
        self.redis.setex(
            f"memory:{memory_id}",
            timedelta(days=365),
            json.dumps(asdict(memory))
        )
        
        return needs_update, new_confidence
    
    def _calculate_reconsidered_confidence(self, temporal_conf: float, consensus: float, contradiction_count: int) -> float:
        """Calculate new confidence based on multiple factors"""
        # Base confidence from temporal decay
        base_conf = temporal_conf
        
        # Consensus boost/penalty
        consensus_factor = 1 + (consensus - 0.5) * 0.5  # Range: 0.75 to 1.25
        
        # Contradiction penalty
        contradiction_penalty = max(0, 1 - (contradiction_count * 0.1))
        
        return min(1.0, base_conf * consensus_factor * contradiction_penalty)
    
    async def _find_related_memories(self, memory_id: str, content: str) -> List[str]:
        """Find memories related to current content - implement with vector search in production"""
        # Simplified implementation - use proper vector similarity in production
        all_memory_keys = self.redis.keys("memory:*")
        related = []
        
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        for key in all_memory_keys[:50]:  # Limit for performance
            if key.decode() == f"memory:{memory_id}":
                continue
                
            memory_data = self.redis.get(key)
            if memory_data:
                memory = MemoryNode(**json.loads(memory_data))
                other_hash = hashlib.md5(memory.content.encode()).hexdigest()
                
                # Simple similarity check
                similarity = sum(c1 == c2 for c1, c2 in zip(content_hash, other_hash)) / len(content_hash)
                if similarity > 0.3:  # Threshold for relatedness
                    related.append(memory.snowflake_id)
        
        return related
    
    async def _detect_contradictions(self, memory: MemoryNode, related_memory_ids: List[str]) -> List[str]:
        """Detect contradictions with related memories"""
        contradictions = []
        
        for related_id in related_memory_ids:
            related_memory = await self._get_memory(related_id)
            if related_memory and self._is_contradictory(memory, related_memory):
                contradictions.append(f"Contradicts {related_id}")
        
        return contradictions
    
    def _is_contradictory(self, memory1: MemoryNode, memory2: MemoryNode) -> bool:
        """Simple contradiction detection - implement with NLP in production"""
        # Placeholder logic - use proper semantic analysis
        negation_words = ["not", "no", "never", "false", "incorrect", "wrong"]
        
        content1_words = memory1.content.lower().split()
        content2_words = memory2.content.lower().split()
        
        # Check for negation patterns (very simplified)
        has_negation = any(word in negation_words for word in content1_words + content2_words)
        return has_negation and len(set(content1_words) & set(content2_words)) > 2
    
    async def _get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """Retrieve memory by ID"""
        memory_data = self.redis.get(f"memory:{memory_id}")
        if memory_data:
            return MemoryNode(**json.loads(memory_data))
        return None
    
    async def run_reconsideration_cycle(self):
        """Run periodic reconsideration on all memories"""
        self.logger.info("Starting reconsideration cycle")
        
        # Get memories that need reconsideration (oldest first)
        memories_to_process = self.redis.zrange("reconsideration_queue", 0, 100)
        
        for memory_id in memories_to_process:
            try:
                needs_update, new_confidence = await self.reconsider_memory(memory_id.decode())
                
                if needs_update:
                    self.logger.info(f"Flagged {memory_id.decode()} for update")
                
                # Remove from queue and re-add with current timestamp for next cycle
                self.redis.zrem("reconsideration_queue", memory_id)
                self.redis.zadd("reconsideration_queue", {memory_id: int(time.time())})
                
            except Exception as e:
                self.logger.error(f"Error reconsidering memory {memory_id}: {e}")
        
        self.logger.info("Reconsideration cycle completed")

# Example usage and testing
async def main():
    """Example usage of the reconsideration engine"""
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    engine = ReconsiderationEngine(redis_client)
    
    # Store some test memories
    memory1 = await engine.store_memory(
        "The sky is blue during the day",
        "general_knowledge",
        0.9
    )
    
    memory2 = await engine.store_memory(
        "The sky is not blue, it appears blue due to light scattering",
        "scientific_explanation",
        0.8
    )
    
    # Simulate time passing and reconsider
    await asyncio.sleep(1)
    
    needs_update1, conf1 = await engine.reconsider_memory(memory1)
    needs_update2, conf2 = await engine.reconsider_memory(memory2)
    
    print(f"Memory 1 needs update: {needs_update1}, confidence: {conf1:.3f}")
    print(f"Memory 2 needs update: {needs_update2}, confidence: {conf2:.3f}")
    
    # Run full reconsideration cycle
    await engine.run_reconsideration_cycle()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
