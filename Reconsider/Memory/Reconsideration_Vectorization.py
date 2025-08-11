# Extended Production Reconsideration System
# Complete implementation with vector embeddings, webhooks, and distributed processing

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import aioredis
import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from celery import Celery
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import asyncpg
import warnings
warnings.filterwarnings("ignore")

# Configuration Management
@dataclass
class SystemConfig:
    """System-wide configuration"""
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://user:pass@localhost/reconsideration_db"
    pinecone_api_key: str = "your-pinecone-key"
    pinecone_environment: str = "production"
    pinecone_index_name: str = "memory-embeddings"
    celery_broker_url: str = "redis://localhost:6379/1"
    embedding_model: str = "all-MiniLM-L6-v2"
    webhook_timeout: int = 30
    reconsideration_batch_size: int = 100
    max_related_memories: int = 50
    consensus_threshold: float = 0.7
    confidence_threshold: float = 0.3
    temporal_decay_rate: float = 0.95

class MemoryStatus(Enum):
    ACTIVE = "active"
    FLAGGED = "flagged"
    DEPRECATED = "deprecated"
    CONFLICTED = "conflicted"
    UPDATING = "updating"

class ContradictionType(Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    FACTUAL = "factual"
    SOURCE_CONFLICT = "source_conflict"

@dataclass
class EnhancedMemoryNode:
    """Enhanced memory node with vector embeddings and metadata"""
    snowflake_id: str
    content: str
    embedding: Optional[List[float]] = None
    confidence_score: float = 0.8
    creation_timestamp: int = field(default_factory=lambda: int(time.time()))
    last_accessed: int = field(default_factory=lambda: int(time.time()))
    last_reconsidered: int = field(default_factory=lambda: int(time.time()))
    access_count: int = 1
    reconsideration_count: int = 0
    consensus_weight: float = 0.5
    status: MemoryStatus = MemoryStatus.ACTIVE
    contradiction_flags: List[Dict[str, Any]] = field(default_factory=list)
    source_context: str = "unknown"
    source_quality_score: float = 0.5
    related_memory_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: int = 1
    parent_memory_id: Optional[str] = None

@dataclass
class WebhookPayload:
    """Payload for webhook notifications"""
    event_type: str
    memory_id: str
    timestamp: int
    data: Dict[str, Any]
    confidence_change: Optional[float] = None

class VectorEmbeddingService:
    """Advanced vector embedding service with multiple model support"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.pinecone_client = None
        self.index = None
        self._init_pinecone()
        
        # Load spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logging.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            pinecone.init(
                api_key=self.config.pinecone_api_key,
                environment=self.config.pinecone_environment
            )
            
            # Create index if it doesn't exist
            if self.config.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    self.config.pinecone_index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.config.pinecone_index_name)
        except Exception as e:
            logging.error(f"Failed to initialize Pinecone: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        # Preprocess text
        if self.nlp:
            doc = self.nlp(text)
            # Remove stop words and keep important tokens
            filtered_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
        else:
            filtered_text = text
        
        # Generate embedding
        embedding = self.model.encode(filtered_text, normalize_embeddings=True)
        return embedding.tolist()
    
    async def store_embedding(self, memory_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Store embedding in vector database"""
        if self.index:
            try:
                self.index.upsert(vectors=[(memory_id, embedding, metadata)])
            except Exception as e:
                logging.error(f"Failed to store embedding for {memory_id}: {e}")
    
    async def find_similar_memories(self, embedding: List[float], top_k: int = 50) -> List[Tuple[str, float]]:
        """Find similar memories using vector search"""
        if not self.index:
            return []
        
        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [(match.id, match.score) for match in results.matches]
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return []
    
    def calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        return float(cosine_similarity([embedding1], [embedding2])[0][0])

class AdvancedContradictionDetector:
    """Advanced contradiction detection using NLP and semantic analysis"""
    
    def __init__(self, embedding_service: VectorEmbeddingService):
        self.embedding_service = embedding_service
        self.nlp = embedding_service.nlp
        
        # Contradiction patterns
        self.negation_patterns = [
            "not", "no", "never", "false", "incorrect", "wrong", "untrue",
            "invalid", "inaccurate", "mistaken", "erroneous"
        ]
        
        self.temporal_indicators = [
            "was", "were", "used to", "previously", "before", "formerly",
            "now", "currently", "today", "recently", "latest"
        ]
    
    async def detect_contradictions(self, memory: EnhancedMemoryNode, related_memories: List[EnhancedMemoryNode]) -> List[Dict[str, Any]]:
        """Detect various types of contradictions"""
        contradictions = []
        
        for related_memory in related_memories:
            # Semantic contradiction detection
            semantic_contradiction = await self._detect_semantic_contradiction(memory, related_memory)
            if semantic_contradiction:
                contradictions.append(semantic_contradiction)
            
            # Temporal contradiction detection
            temporal_contradiction = await self._detect_temporal_contradiction(memory, related_memory)
            if temporal_contradiction:
                contradictions.append(temporal_contradiction)
            
            # Factual contradiction detection
            factual_contradiction = await self._detect_factual_contradiction(memory, related_memory)
            if factual_contradiction:
                contradictions.append(factual_contradiction)
        
        return contradictions
    
    async def _detect_semantic_contradiction(self, memory1: EnhancedMemoryNode, memory2: EnhancedMemoryNode) -> Optional[Dict[str, Any]]:
        """Detect semantic contradictions using NLP"""
        if not self.nlp or not memory1.embedding or not memory2.embedding:
            return None
        
        # High similarity but low confidence suggests contradiction
        similarity = self.embedding_service.calculate_semantic_similarity(memory1.embedding, memory2.embedding)
        
        if similarity > 0.8:  # Very similar content
            # Check for negation patterns
            doc1 = self.nlp(memory1.content.lower())
            doc2 = self.nlp(memory2.content.lower())
            
            negations_1 = sum(1 for token in doc1 if token.text in self.negation_patterns)
            negations_2 = sum(1 for token in doc2 if token.text in self.negation_patterns)
            
            if abs(negations_1 - negations_2) > 0:  # Different negation patterns
                return {
                    "type": ContradictionType.SEMANTIC.value,
                    "contradicting_memory": memory2.snowflake_id,
                    "similarity_score": similarity,
                    "description": f"Semantic contradiction detected with {memory2.snowflake_id}",
                    "confidence": min(memory1.confidence_score, memory2.confidence_score)
                }
        
        return None
    
    async def _detect_temporal_contradiction(self, memory1: EnhancedMemoryNode, memory2: EnhancedMemoryNode) -> Optional[Dict[str, Any]]:
        """Detect temporal contradictions"""
        # Check for conflicting temporal indicators
        content1_lower = memory1.content.lower()
        content2_lower = memory2.content.lower()
        
        temporal_1 = [indicator for indicator in self.temporal_indicators if indicator in content1_lower]
        temporal_2 = [indicator for indicator in self.temporal_indicators if indicator in content2_lower]
        
        if temporal_1 and temporal_2:
            # Check for contradictory temporal patterns
            past_indicators = ["was", "were", "used to", "previously", "before", "formerly"]
            present_indicators = ["now", "currently", "today", "recently", "latest"]
            
            has_past_1 = any(indicator in temporal_1 for indicator in past_indicators)
            has_present_1 = any(indicator in temporal_1 for indicator in present_indicators)
            has_past_2 = any(indicator in temporal_2 for indicator in past_indicators)
            has_present_2 = any(indicator in temporal_2 for indicator in present_indicators)
            
            if (has_past_1 and has_present_2) or (has_present_1 and has_past_2):
                return {
                    "type": ContradictionType.TEMPORAL.value,
                    "contradicting_memory": memory2.snowflake_id,
                    "description": f"Temporal contradiction detected with {memory2.snowflake_id}",
                    "temporal_indicators_1": temporal_1,
                    "temporal_indicators_2": temporal_2
                }
        
        return None
    
    async def _detect_factual_contradiction(self, memory1: EnhancedMemoryNode, memory2: EnhancedMemoryNode) -> Optional[Dict[str, Any]]:
        """Detect factual contradictions using named entity recognition"""
        if not self.nlp:
            return None
        
        doc1 = self.nlp(memory1.content)
        doc2 = self.nlp(memory2.content)
        
        # Extract named entities
        entities_1 = {ent.text.lower(): ent.label_ for ent in doc1.ents}
        entities_2 = {ent.text.lower(): ent.label_ for ent in doc2.ents}
        
        # Check for conflicting facts about same entities
        common_entities = set(entities_1.keys()) & set(entities_2.keys())
        
        for entity in common_entities:
            if entities_1[entity] == entities_2[entity]:  # Same entity type
                # Check if the contexts suggest contradiction
                # This is a simplified check - in production, use more sophisticated fact checking
                if "not" in memory1.content.lower() or "not" in memory2.content.lower():
                    return {
                        "type": ContradictionType.FACTUAL.value,
                        "contradicting_memory": memory2.snowflake_id,
                        "conflicting_entity": entity,
                        "entity_type": entities_1[entity],
                        "description": f"Factual contradiction about {entity}"
                    }
        
        return None

class WebhookManager:
    """Manages webhook notifications for memory events"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.session = None
        self.webhooks: Dict[str, str] = {}  # event_type -> webhook_url
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def register_webhook(self, event_type: str, webhook_url: str):
        """Register a webhook for specific event types"""
        self.webhooks[event_type] = webhook_url
    
    async def notify_webhook(self, payload: WebhookPayload):
        """Send webhook notification"""
        if payload.event_type not in self.webhooks:
            return
        
        webhook_url = self.webhooks[payload.event_type]
        
        try:
            if self.session:
                async with self.session.post(
                    webhook_url,
                    json=asdict(payload),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logging.info(f"Webhook notification sent successfully: {payload.event_type}")
                    else:
                        logging.warning(f"Webhook notification failed: {response.status}")
        except Exception as e:
            logging.error(f"Webhook notification error: {e}")

class EnhancedReconsiderationEngine:
    """Enhanced reconsideration engine with full production features"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.redis = None
        self.postgres_pool = None
        self.snowflake_gen = SnowflakeGenerator()
        self.embedding_service = VectorEmbeddingService(config)
        self.contradiction_detector = AdvancedContradictionDetector(self.embedding_service)
        self.webhook_manager = None
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.metrics = {
            "memories_processed": 0,
            "contradictions_detected": 0,
            "updates_triggered": 0,
            "consensus_changes": 0
        }
    
    async def initialize(self):
        """Initialize all services"""
        # Initialize Redis
        self.redis = await aioredis.from_url(self.config.redis_url)
        
        # Initialize PostgreSQL pool
        self.postgres_pool = await asyncpg.create_pool(self.config.postgres_url)
        
        # Initialize webhook manager
        self.webhook_manager = WebhookManager(self.config)
        await self.webhook_manager.__aenter__()
        
        # Create database tables
        await self._create_tables()
        
        self.logger.info("Enhanced Reconsideration Engine initialized")
    
    async def _create_tables(self):
        """Create PostgreSQL tables for metadata storage"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    snowflake_id VARCHAR(50) PRIMARY KEY,
                    creation_timestamp BIGINT NOT NULL,
                    last_accessed BIGINT NOT NULL,
                    last_reconsidered BIGINT NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    reconsideration_count INTEGER DEFAULT 0,
                    confidence_score FLOAT DEFAULT 0.8,
                    consensus_weight FLOAT DEFAULT 0.5,
                    status VARCHAR(20) DEFAULT 'active',
                    source_context VARCHAR(255),
                    source_quality_score FLOAT DEFAULT 0.5,
                    version INTEGER DEFAULT 1,
                    parent_memory_id VARCHAR(50),
                    tags TEXT[],
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS memory_contradictions (
                    id SERIAL PRIMARY KEY,
                    memory_id VARCHAR(50) NOT NULL,
                    contradicting_memory_id VARCHAR(50) NOT NULL,
                    contradiction_type VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    description TEXT,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE
                );
                
                CREATE TABLE IF NOT EXISTS reconsideration_history (
                    id SERIAL PRIMARY KEY,
                    memory_id VARCHAR(50) NOT NULL,
                    old_confidence FLOAT NOT NULL,
                    new_confidence FLOAT NOT NULL,
                    consensus_change FLOAT,
                    contradictions_found INTEGER DEFAULT 0,
                    action_taken VARCHAR(100),
                    processed_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_memory_confidence ON memory_metadata(confidence_score);
                CREATE INDEX IF NOT EXISTS idx_memory_status ON memory_metadata(status);
                CREATE INDEX IF NOT EXISTS idx_contradiction_unresolved ON memory_contradictions(resolved) WHERE resolved = FALSE;
            """)
    
    async def store_enhanced_memory(self, content: str, source_context: str, 
                                  initial_confidence: float = 0.8, 
                                  tags: List[str] = None) -> str:
        """Store memory with full enhancement pipeline"""
        snowflake_id = self.snowflake_gen.generate()
        current_time = int(time.time())
        
        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(content)
        
        # Create enhanced memory node
        memory = EnhancedMemoryNode(
            snowflake_id=snowflake_id,
            content=content,
            embedding=embedding,
            confidence_score=initial_confidence,
            source_context=source_context,
            tags=tags or []
        )
        
        # Store in Redis (fast access)
        await self.redis.setex(
            f"memory:{snowflake_id}",
            int(timedelta(days=365).total_seconds()),
            json.dumps(asdict(memory), default=str)
        )
        
        # Store metadata in PostgreSQL (persistent, queryable)
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory_metadata (
                    snowflake_id, creation_timestamp, last_accessed, last_reconsidered,
                    confidence_score, source_context, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, snowflake_id, current_time, current_time, current_time,
                initial_confidence, source_context, tags or [])
        
        # Store embedding in vector database
        await self.embedding_service.store_embedding(
            snowflake_id, embedding, 
            {"content": content, "confidence": initial_confidence, "source": source_context}
        )
        
        # Add to reconsideration queue
        await self.redis.zadd("reconsideration_queue", {snowflake_id: current_time})
        
        # Send webhook notification
        if self.webhook_manager:
            await self.webhook_manager.notify_webhook(WebhookPayload(
                event_type="memory_created",
                memory_id=snowflake_id,
                timestamp=current_time,
                data={"confidence": initial_confidence, "source": source_context}
            ))
        
        self.logger.info(f"Enhanced memory stored: {snowflake_id}")
        return snowflake_id
    
    async def enhanced_reconsideration(self, memory_id: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Enhanced reconsideration with full analysis pipeline"""
        memory = await self._get_enhanced_memory(memory_id)
        if not memory:
            return False, 0.0, {"error": "Memory not found"}
        
        old_confidence = memory.confidence_score
        analysis_results = {}
        
        # Find related memories using vector search
        if memory.embedding:
            similar_memories = await self.embedding_service.find_similar_memories(
                memory.embedding, self.config.max_related_memories
            )
            related_memory_objects = []
            
            for related_id, similarity_score in similar_memories:
                if related_id != memory_id:
                    related_memory = await self._get_enhanced_memory(related_id)
                    if related_memory:
                        related_memory_objects.append(related_memory)
            
            analysis_results["related_memories_found"] = len(related_memory_objects)
        else:
            related_memory_objects = []
            analysis_results["related_memories_found"] = 0
        
        # Detect contradictions
        contradictions = await self.contradiction_detector.detect_contradictions(
            memory, related_memory_objects
        )
        analysis_results["contradictions_detected"] = len(contradictions)
        
        # Store contradictions in database
        for contradiction in contradictions:
            await self._store_contradiction(memory_id, contradiction)
        
        # Calculate consensus
        consensus_score = await self._calculate_enhanced_consensus(memory, related_memory_objects)
        analysis_results["consensus_score"] = consensus_score
        
        # Apply temporal decay
        time_elapsed = int(time.time()) - memory.creation_timestamp
        days_elapsed = time_elapsed / (24 * 3600)
        temporal_confidence = memory.confidence_score * (self.config.temporal_decay_rate ** days_elapsed)
        analysis_results["temporal_decay_applied"] = temporal_confidence
        
        # Calculate new confidence
        new_confidence = self._calculate_enhanced_confidence(
            temporal_confidence, consensus_score, len(contradictions), memory.source_quality_score
        )
        
        # Update memory status based on confidence
        new_status = self._determine_memory_status(new_confidence, len(contradictions))
        
        # Update memory
        memory.confidence_score = new_confidence
        memory.consensus_weight = consensus_score
        memory.contradiction_flags = contradictions
        memory.status = new_status
        memory.last_reconsidered = int(time.time())
        memory.reconsideration_count += 1
        memory.related_memory_ids = [rm.snowflake_id for rm in related_memory_objects[:10]]  # Top 10
        
        # Store updated memory
        await self._store_updated_memory(memory)
        
        # Record reconsideration history
        await self._record_reconsideration_history(
            memory_id, old_confidence, new_confidence, consensus_score, len(contradictions)
        )
        
        # Determine if memory needs updating
        needs_update = new_confidence < self.config.confidence_threshold
        action_taken = "flagged_for_update" if needs_update else "confidence_updated"
        
        if needs_update:
            await self.redis.sadd("memories_to_update", memory_id)
            self.metrics["updates_triggered"] += 1
        
        # Send webhook notification
        if self.webhook_manager:
            await self.webhook_manager.notify_webhook(WebhookPayload(
                event_type="memory_reconsidered",
                memory_id=memory_id,
                timestamp=int(time.time()),
                data=analysis_results,
                confidence_change=new_confidence - old_confidence
            ))
        
        # Update metrics
        self.metrics["memories_processed"] += 1
        self.metrics["contradictions_detected"] += len(contradictions)
        if abs(consensus_score - memory.consensus_weight) > 0.1:
            self.metrics["consensus_changes"] += 1
        
        analysis_results.update({
            "needs_update": needs_update,
            "new_confidence": new_confidence,
            "confidence_change": new_confidence - old_confidence,
            "action_taken": action_taken,
            "new_status": new_status.value
        })
        
        return needs_update, new_confidence, analysis_results
    
    def _calculate_enhanced_confidence(self, temporal_conf: float, consensus: float, 
                                     contradiction_count: int, source_quality: float) -> float:
        """Enhanced confidence calculation with multiple factors"""
        base_conf = temporal_conf
        
        # Consensus factor (stronger influence)
        consensus_factor = 0.5 + (consensus * 0.8)  # Range: 0.5 to 1.3
        
        # Contradiction penalty (exponential)
        contradiction_penalty = max(0.1, 0.9 ** contradiction_count)
        
        # Source quality boost
        source_factor = 0.8 + (source_quality * 0.4)  # Range: 0.8 to 1.2
        
        # Access frequency bonus (more accessed = more validated)
        # This would be calculated based on access patterns
        
        new_confidence = min(1.0, base_conf * consensus_factor * contradiction_penalty * source_factor)
        return max(0.0, new_confidence)
    
    def _determine_memory_status(self, confidence: float, contradiction_count: int) -> MemoryStatus:
        """Determine memory status based on confidence and contradictions"""
        if contradiction_count > 3:
            return MemoryStatus.CONFLICTED
        elif confidence < 0.2:
            return MemoryStatus.DEPRECATED
        elif confidence < self.config.confidence_threshold:
            return MemoryStatus.FLAGGED
        elif contradiction_count > 0:
            return MemoryStatus.UPDATING
        else:
            return MemoryStatus.ACTIVE
    
    async def _calculate_enhanced_consensus(self, memory: EnhancedMemoryNode, 
                                          related_memories: List[EnhancedMemoryNode]) -> float:
        """Calculate consensus using vector similarity and confidence weighting"""
        if not related_memories or not memory.embedding:
            return 0.5
        
        consensus_scores = []
        total_weight = 0
        
        for related_memory in related_memories:
            if related_memory.embedding:
                # Calculate semantic similarity
                similarity = self.embedding_service.calculate_semantic_similarity(
                    memory.embedding, related_memory.embedding
                )
                
                # Weight by related memory confidence and age
                age_factor = max(0.1, 1.0 - ((int(time.time()) - related_memory.creation_timestamp) / (365 * 24 * 3600)))
                weight = related_memory.confidence_score * age_factor
                
                consensus_scores.append(similarity * weight)
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return sum(consensus_scores) / total_weight
    
    async def _store_contradiction(self, memory_id: str, contradiction: Dict[str, Any]):
        """Store contradiction in database"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory_contradictions (
                    memory_id, contradicting_memory_id, contradiction_type, 
                    confidence, description
                ) VALUES ($1, $2, $3, $4, $5)
            """, memory_id, contradiction.get("contradicting_memory"),
                contradiction.get("type"), contradiction.get("confidence", 0.5),
                contradiction.get("description", ""))
    
    async def _record_reconsideration_history(self, memory_id: str, old_confidence: float,
                                            new_confidence: float, consensus_change: float,
                                            contradictions_found: int):
        """Record reconsideration history for analytics"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO reconsideration_history (
                    memory_id, old_confidence, new_confidence, consensus_change,
                    contradictions_found, action_taken
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, memory_id, old_confidence, new_confidence, consensus_change,
                contradictions_found, "reconsidered")
    
    async def _get_enhanced_memory(self, memory_id: str) -> Optional[EnhancedMemoryNode]:
        """Retrieve enhanced memory from Redis"""
        memory_data = await self.redis.get(f"memory:{memory_id}")
        if memory_data:
            data = json.loads(memory_data)
            return EnhancedMemoryNode(**data)
        return None
    
    async def _store_updated_memory(self, memory: EnhancedMemoryNode):
        """Store updated memory in both Redis and PostgreSQL"""
        # Redis
        await self.redis.setex(
            f"memory:{memory.snowflake_id}",
            int(timedelta(days=365).total_seconds()),
            json.dumps(asdict(memory), default=str)
        )
        
        # PostgreSQL
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE memory_metadata SET
                    last_reconsidered = $1, reconsideration_count = $2,
                    confidence_score = $3, consensus_weight = $4,
                    status = $5, updated_at = NOW()
                WHERE snowflake_id = $6
            """, memory.last_reconsidered, memory.reconsideration_count,
                memory.confidence_score, memory.consensus_weight,
                memory.status.value, memory.snowflake_id)
    
    async def run_distributed_reconsideration_cycle(self):
        """Run distributed reconsideration using Celery for heavy processing"""
        self.logger.info("Starting distributed reconsideration cycle")
        
        # Get batch of memories to process
        memories_to_process = await self.redis.zrange(
            "reconsideration_queue", 0, self.config.reconsideration_batch_size - 1
        )
        
        # Process in batches
        tasks = []
        for memory_id in memories_to_process:
            # Queue Celery task for processing
            task = process_memory_reconsideration.delay(memory_id.decode())
            tasks.append(task)
        
        # Wait for completion and collect results
        results = []
        for task in tasks:
            try:
                result = task.get(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task failed: {e}")
        
        self.logger.info(f"Processed {len(results)} memories in distributed cycle")
        return results
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        # Memory statistics
        total_memories = await self.redis.dbsize()
        flagged_memories = await self.redis.scard("memories_to_update")
        
        # Database statistics
        async with self.postgres_pool.acquire() as conn:
            db_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_memories,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(*) FILTER (WHERE status = 'flagged') as flagged_count,
                    COUNT(*) FILTER (WHERE status = 'conflicted') as conflicted_count,
                    (SELECT COUNT(*) FROM memory_contradictions WHERE resolved = FALSE) as unresolved_contradictions
                FROM memory_metadata
            """)
        
        return {
            "runtime_metrics": self.metrics,
            "storage_metrics": {
                "total_memories_redis": total_memories,
                "flagged_for_update": flagged_memories,
                "total_memories_db": db_stats["total_memories"],
                "average_confidence": float(db_stats["avg_confidence"] or 0),
                "flagged_count": db_stats["flagged_count"],
                "conflicted_count": db_stats["conflicted_count"],
                "unresolved_contradictions": db_stats["unresolved_contradictions"]
            },
            "vector_db_metrics": {
                "index_name": self.config.pinecone_index_name,
                "embedding_dimension": 384
            }
        }

# Celery configuration for distributed processing
celery_app = Celery('reconsideration_worker', broker='redis://localhost:6379/1')

@celery_app.task
def process_memory_reconsideration(memory_id: str) -> Dict[str, Any]:
    """Celery task for processing memory reconsideration"""
    # This would be called by the distributed system
    # Implementation would initialize engine and call enhanced_reconsideration
    pass

# FastAPI application for REST API
app = FastAPI(title="Enhanced Reconsideration System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[EnhancedReconsiderationEngine] = None

@app.on_event("startup")
async def startup_event():
    global engine
    config = SystemConfig()
    engine = EnhancedReconsiderationEngine(config)
    await engine.initialize()

class MemoryCreateRequest(BaseModel):
    content: str
    source_context: str
    initial_confidence: float = 0.8
    tags: Optional[List[str]] = None

class WebhookRegisterRequest(BaseModel):
    event_type: str
    webhook_url: str

@app.post("/api/v2/memory")
async def create_memory(request: MemoryCreateRequest):
    """Create a new memory with enhanced processing"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    memory_id = await engine.store_enhanced_memory(
        request.content, request.source_context, 
        request.initial_confidence, request.tags
    )
    
    return {"memory_id": memory_id, "status": "created"}

@app.post("/api/v2/memory/{memory_id}/reconsider")
async def reconsider_memory(memory_id: str, background_tasks: BackgroundTasks):
    """Trigger reconsideration for a specific memory"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    needs_update, confidence, analysis = await engine.enhanced_reconsideration(memory_id)
    
    return {
        "memory_id": memory_id,
        "needs_update": needs_update,
        "new_confidence": confidence,
        "analysis": analysis
    }

@app.get("/api/v2/memory/{memory_id}")
async def get_memory(memory_id: str):
    """Retrieve memory details"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    memory = await engine._get_enhanced_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return asdict(memory)

@app.get("/api/v2/metrics")
async def get_metrics():
    """Get system performance metrics"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    return await engine.get_system_metrics()

@app.post("/api/v2/webhooks/register")
async def register_webhook(request: WebhookRegisterRequest):
    """Register a webhook for events"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    engine.webhook_manager.register_webhook(request.event_type, request.webhook_url)
    return {"status": "webhook_registered", "event_type": request.event_type}

@app.post("/api/v2/reconsideration/cycle")
async def trigger_reconsideration_cycle(background_tasks: BackgroundTasks):
    """Trigger a full reconsideration cycle"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    background_tasks.add_task(engine.run_distributed_reconsideration_cycle)
    return {"status": "cycle_triggered"}

@app.websocket("/api/v2/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    
    try:
        while True:
            if engine:
                metrics = await engine.get_system_metrics()
                await websocket.send_json(metrics)
            await asyncio.sleep(5)  # Send metrics every 5 seconds
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
