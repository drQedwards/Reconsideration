# test_suite.py - Comprehensive test suite for the reconsideration system
import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

import aioredis
import asyncpg
from fastapi.testclient import TestClient
from httpx import AsyncClient

from main import (
    EnhancedReconsiderationEngine, VectorEmbeddingService, 
    AdvancedContradictionDetector, WebhookManager, SystemConfig,
    EnhancedMemoryNode, MemoryStatus, app
)

# Test Configuration
TEST_CONFIG = SystemConfig(
    redis_url="redis://localhost:6379/15",  # Use different DB for tests
    postgres_url="postgresql://test_user:test_pass@localhost/test_reconsideration_db",
    pinecone_api_key="test-key",
    pinecone_environment="test",
    pinecone_index_name="test-memory-embeddings"
)

@pytest.fixture
async def redis_client():
    """Redis client fixture for testing"""
    client = await aioredis.from_url(TEST_CONFIG.redis_url)
    await client.flushdb()  # Clean test database
    yield client
    await client.flushdb()
    await client.close()

@pytest.fixture
async def postgres_pool():
    """PostgreSQL connection pool for testing"""
    pool = await asyncpg.create_pool(TEST_CONFIG.postgres_url)
    
    # Create test tables
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS memory_contradictions, reconsideration_history, memory_metadata CASCADE")
        # Create tables (use same schema as main system)
        await conn.execute("""
            CREATE TABLE memory_metadata (
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
        """)
    
    yield pool
    await pool.close()

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing"""
    service = Mock(spec=VectorEmbeddingService)
    service.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    service.store_embedding = AsyncMock()
    service.find_similar_memories = AsyncMock(return_value=[("test_id_1", 0.8), ("test_id_2", 0.6)])
    service.calculate_semantic_similarity = Mock(return_value=0.75)
    return service

@pytest.fixture
async def test_engine(redis_client, postgres_pool, mock_embedding_service):
    """Test engine fixture"""
    engine = EnhancedReconsiderationEngine(TEST_CONFIG)
    engine.redis = redis_client
    engine.postgres_pool = postgres_pool
    engine.embedding_service = mock_embedding_service
    
    # Mock webhook manager
    engine.webhook_manager = Mock(spec=WebhookManager)
    engine.webhook_manager.notify_webhook = AsyncMock()
    
    yield engine

class TestSnowflakeGenerator:
    """Test snowflake ID generation"""
    
    def test_snowflake_generation(self):
        from main import SnowflakeGenerator
        
        gen = SnowflakeGenerator(worker_id=1, datacenter_id=1)
        
        # Generate multiple IDs
        ids = [gen.generate() for _ in range(1000)]
        
        # Check uniqueness
        assert len(set(ids)) == 1000
        
        # Check format (hex string)
        for id_str in ids:
            assert id_str.startswith('0x')
            int(id_str, 16)  # Should not raise exception

class TestVectorEmbeddingService:
    """Test vector embedding functionality"""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        service = VectorEmbeddingService(TEST_CONFIG)
        
        text = "The sky is blue during the day"
        embedding = await service.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Expected dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_semantic_similarity(self):
        service = VectorEmbeddingService(TEST_CONFIG)
        
        # Create test embeddings
        embedding1 = [0.1] * 384
        embedding2 = [0.2] * 384
        
        similarity = service.calculate_semantic_similarity(embedding1, embedding2)
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

class TestAdvancedContradictionDetector:
    """Test contradiction detection functionality"""
    
    @pytest.fixture
    def detector(self, mock_embedding_service):
        return AdvancedContradictionDetector(mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_semantic_contradiction_detection(self, detector):
        memory1 = EnhancedMemoryNode(
            snowflake_id="test_id_1",
            content="The sky is blue",
            embedding=[0.1] * 384,
            confidence_score=0.8
        )
        
        memory2 = EnhancedMemoryNode(
            snowflake_id="test_id_2", 
            content="The sky is not blue",
            embedding=[0.1] * 384,  # Similar embedding
            confidence_score=0.7
        )
        
        contradictions = await detector.detect_contradictions(memory1, [memory2])
        
        assert len(contradictions) > 0
        assert contradictions[0]["type"] == "semantic"
    
    @pytest.mark.asyncio
    async def test_temporal_contradiction_detection(self, detector):
        memory1 = EnhancedMemoryNode(
            snowflake_id="test_id_1",
            content="Paris was the capital of France",
            embedding=[0.1] * 384,
            confidence_score=0.8
        )
        
        memory2 = EnhancedMemoryNode(
            snowflake_id="test_id_2",
            content="Paris is currently the capital of France",
            embedding=[0.1] * 384,
            confidence_score=0.9
        )
        
        contradictions = await detector.detect_contradictions(memory1, [memory2])
        
        # Should detect temporal contradiction
        temporal_contradictions = [c for c in contradictions if c["type"] == "temporal"]
        assert len(temporal_contradictions) > 0

class TestEnhancedReconsiderationEngine:
    """Test main reconsideration engine functionality"""
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, test_engine):
        content = "Test memory content"
        source_context = "unit_test"
        tags = ["test", "memory"]
        
        memory_id = await test_engine.store_enhanced_memory(
            content, source_context, tags=tags
        )
        
        assert memory_id is not None
        assert memory_id.startswith('0x')
        
        # Verify storage in Redis
        stored_data = await test_engine.redis.get(f"memory:{memory_id}")
        assert stored_data is not None
        
        memory_dict = json.loads(stored_data)
        assert memory_dict["content"] == content
        assert memory_dict["source_context"] == source_context
        assert memory_dict["tags"] == tags
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, test_engine):
        # Store a memory first
        memory_id = await test_engine.store_enhanced_memory(
            "Test content", "test_context"
        )
        
        # Retrieve it
        memory = await test_engine._get_enhanced_memory(memory_id)
        
        assert memory is not None
        assert memory.snowflake_id == memory_id
        assert memory.content == "Test content"
        assert memory.source_context == "test_context"
    
    @pytest.mark.asyncio
    async def test_enhanced_reconsideration(self, test_engine):
        # Store a test memory
        memory_id = await test_engine.store_enhanced_memory(
            "Test memory for reconsideration", "test_context", 0.8
        )
        
        # Run reconsideration
        needs_update, new_confidence, analysis = await test_engine.enhanced_reconsideration(memory_id)
        
        assert isinstance(needs_update, bool)
        assert isinstance(new_confidence, float)
        assert 0 <= new_confidence <= 1
        assert isinstance(analysis, dict)
        assert "related_memories_found" in analysis
        assert "contradictions_detected" in analysis
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, test_engine):
        # Test confidence calculation with various inputs
        temporal_conf = 0.8
        consensus = 0.6
        contradiction_count = 1
        source_quality = 0.7
        
        new_confidence = test_engine._calculate_enhanced_confidence(
            temporal_conf, consensus, contradiction_count, source_quality
        )
        
        assert 0 <= new_confidence <= 1
        assert isinstance(new_confidence, float)
    
    @pytest.mark.asyncio
    async def test_memory_status_determination(self, test_engine):
        # Test different status determinations
        status1 = test_engine._determine_memory_status(0.9, 0)  # High confidence, no contradictions
        assert status1 == MemoryStatus.ACTIVE
        
        status2 = test_engine._determine_memory_status(0.1, 0)  # Low confidence
        assert status2 == MemoryStatus.DEPRECATED
        
        status3 = test_engine._determine_memory_status(0.5, 5)  # Many contradictions
        assert status3 == MemoryStatus.CONFLICTED

class TestWebhookManager:
    """Test webhook functionality"""
    
    @pytest.mark.asyncio
    async def test_webhook_registration_and_notification(self):
        from main import WebhookPayload
        
        config = TEST_CONFIG
        
        async with WebhookManager(config) as webhook_manager:
            # Register a webhook
            webhook_manager.register_webhook("test_event", "http://localhost:8080/webhook")
            
            # Create test payload
            payload = WebhookPayload(
                event_type="test_event",
                memory_id="test_memory_id",
                timestamp=int(time.time()),
                data={"test": "data"}
            )
            
            # Mock the HTTP session
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = Mock()
                mock_response.status = 200
                mock_post.return_value.__aenter__.return_value = mock_response
                
                await webhook_manager.notify_webhook(payload)
                
                # Verify the webhook was called
                mock_post.assert_called_once()

class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_create_memory_endpoint(self):
        client = TestClient(app)
        
        # Mock the engine
        with patch('main.engine') as mock_engine:
            mock_engine.store_enhanced_memory = AsyncMock(return_value="test_memory_id")
            
            response = client.post("/api/v2/memory", json={
                "content": "Test memory content",
                "source_context": "api_test",
                "initial_confidence": 0.8,
                "tags": ["test"]
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["memory_id"] == "test_memory_id"
            assert data["status"] == "created"
    
    def test_reconsider_memory_endpoint(self):
        client = TestClient(app)
        
        with patch('main.engine') as mock_engine:
            mock_engine.enhanced_reconsideration = AsyncMock(
                return_value=(True, 0.6, {"analysis": "complete"})
            )
            
            response = client.post("/api/v2/memory/test_id/reconsider")
            
            assert response.status_code == 200
            data = response.json()
            assert data["needs_update"] is True
            assert data["new_confidence"] == 0.6
    
    def test_get_memory_endpoint(self):
        client = TestClient(app)
        
        with patch('main.engine') as mock_engine:
            mock_memory = EnhancedMemoryNode(
                snowflake_id="test_id",
                content="Test content",
                confidence_score=0.8
            )
            mock_engine._get_enhanced_memory = AsyncMock(return_value=mock_memory)
            
            response = client.get("/api/v2/memory/test_id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["snowflake_id"] == "test_id"
            assert data["content"] == "Test content"
    
    def test_metrics_endpoint(self):
        client = TestClient(app)
        
        with patch('main.engine') as mock_engine:
            mock_engine.get_system_metrics = AsyncMock(return_value={
                "runtime_metrics": {"memories_processed": 100},
                "storage_metrics": {"total_memories_redis": 50}
            })
            
            response = client.get("/api/v2/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "runtime_metrics" in data
            assert "storage_metrics" in data

class TestPerformanceAndLoad:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_storage(self, test_engine):
        """Test concurrent memory storage operations"""
        async def store_memory(i):
            return await test_engine.store_enhanced_memory(
                f"Test memory {i}", "load_test", 0.8
            )
        
        # Store 100 memories concurrently
        tasks = [store_memory(i) for i in range(100)]
        memory_ids = await asyncio.gather(*tasks)
        
        assert len(memory_ids) == 100
        assert len(set(memory_ids)) == 100  # All unique
    
    @pytest.mark.asyncio
    async def test_batch_reconsideration_performance(self, test_engine):
        """Test batch reconsideration performance"""
        # Store test memories
        memory_ids = []
        for i in range(50):
            memory_id = await test_engine.store_enhanced_memory(
                f"Performance test memory {i}", "perf_test", 0.8
            )
            memory_ids.append(memory_id)
        
        # Time batch reconsideration
        start_time = time.time()
        
        tasks = [test_engine.enhanced_reconsideration(mid) for mid in memory_ids]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 50 memories in reasonable time (< 30 seconds for test)
        assert processing_time < 30
        assert len(results) == 50

class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, test_engine):
        """Test complete memory lifecycle from creation to reconsideration"""
        
        # 1. Create initial memory
        memory_id = await test_engine.store_enhanced_memory(
            "The Earth is flat", "controversial_source", 0.3
        )
        
        # 2. Create contradicting memory
        contradicting_id = await test_engine.store_enhanced_memory(
            "The Earth is round", "scientific_source", 0.9
        )
        
        # 3. Reconsider the first memory
        needs_update, new_confidence, analysis = await test_engine.enhanced_reconsideration(memory_id)
        
        # 4. Verify the low-confidence memory is flagged
        assert needs_update or new_confidence < 0.5
        assert "contradictions_detected" in analysis
        
        # 5. Check memory status
        memory = await test_engine._get_enhanced_memory(memory_id)
        assert memory.status in [MemoryStatus.FLAGGED, MemoryStatus.DEPRECATED, MemoryStatus.CONFLICTED]
    
    @pytest.mark.asyncio
    async def test_consensus_building_scenario(self, test_engine):
        """Test consensus building with multiple related memories"""
        
        # Create multiple memories about the same topic
        base_content = "Python is a programming language"
        
        memory_ids = []
        for i, confidence in enumerate([0.8, 0.9, 0.85, 0.7, 0.95]):
            memory_id = await test_engine.store_enhanced_memory(
                f"{base_content} used for {['web', 'AI', 'data', 'automation', 'science'][i]}",
                f"source_{i}", confidence
            )
            memory_ids.append(memory_id)
        
        # Reconsider each memory
        for memory_id in memory_ids:
            needs_update, new_confidence, analysis = await test_engine.enhanced_reconsideration(memory_id)
            
            # High consensus should boost confidence
            assert analysis["consensus_score"] > 0.5
            assert not needs_update  # Should not need updates due to good consensus

# Load Testing with Locust
"""
# locustfile.py - Load testing configuration

from locust import HttpUser, task, between
import json
import random

class ReconsiderationUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Create some test memories
        self.memory_ids = []
        for i in range(5):
            response = self.client.post("/api/v2/memory", json={
                "content": f"Load test memory {i} {random.randint(1, 1000)}",
                "source_context": "load_test",
                "initial_confidence": random.uniform(0.5, 1.0)
            })
            if response.status_code == 200:
                self.memory_ids.append(response.json()["memory_id"])
    
    @task(3)
    def create_memory(self):
        self.client.post("/api/v2/memory", json={
            "content": f"Random memory {random.randint(1, 10000)}",
            "source_context": "load_test",
            "initial_confidence": random.uniform(0.3, 1.0),
            "tags": [f"tag{random.randint(1, 5)}"]
        })
    
    @task(2)
    def reconsider_memory(self):
        if self.memory_ids:
            memory_id = random.choice(self.memory_ids)
            self.client.post(f"/api/v2/memory/{memory_id}/reconsider")
    
    @task(1)
    def get_memory(self):
        if self.memory_ids:
            memory_id = random.choice(self.memory_ids)
            self.client.get(f"/api/v2/memory/{memory_id}")
    
    @task(1)
    def get_metrics(self):
        self.client.get("/api/v2/metrics")

# Run with: locust -f locustfile.py --host=http://localhost:8000
"""

# Pytest configuration
PYTEST_CONFIG = """
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --asyncio-mode=auto
    --cov=main
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=85
asyncio_mode = auto

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
"""

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

---

# deploy.sh - Deployment script
#!/bin/bash
set -e

echo "🚀 Starting Enhanced Reconsideration System Deployment"

# Configuration
PROJECT_NAME="reconsideration-system"
VERSION=${1:-"latest"}
ENVIRONMENT=${2:-"production"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if environment file exists
    if [ ! -f ".env" ]; then
        log_warn ".env file not found. Creating from template..."
        cp .env.example .env
        log_warn "Please update .env file with your configuration before proceeding"
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build API image
    docker build -f Dockerfile.api -t ${PROJECT_NAME}-api:${VERSION} .
    
    # Build worker image
    docker build -f Dockerfile.worker -t ${PROJECT_NAME}-worker:${VERSION} .
    
    log_info "Docker images built successfully ✓"
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    # Start test dependencies
    docker-compose -f docker-compose.test.yml up -d redis postgres
    
    # Wait for services to be ready
    sleep 10
    
    # Run tests
    python -m pytest tests/ -v --asyncio-mode=auto --cov=main --cov-report=term-missing
    
    if [ $? -eq 0 ]; then
        log_info "All tests passed ✓"
    else
        log_error "Tests failed ✗"
        docker-compose -f docker-compose.test.yml down
        exit 1
    fi
    
    # Cleanup test environment
    docker-compose -f docker-compose.test.yml down
}

# Deploy to environment
deploy() {
    log_info "Deploying to ${ENVIRONMENT} environment..."
    
    # Stop existing containers
    docker-compose down
    
    # Pull latest images (if not building locally)
    if [ "${VERSION}" != "latest" ]; then
        docker-compose pull
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Health check
    health_check
    
    log_info "Deployment completed successfully ✓"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    # Check API health
    if curl -f http://localhost:8000/api/v2/metrics > /dev/null 2>&1; then
        log_info "API health check passed ✓"
    else
        log_error "API health check failed ✗"
        exit 1
    fi
    
    # Check Redis
    if docker-compose exec redis redis-cli ping | grep -q PONG; then
        log_info "Redis health check passed ✓"
    else
        log_error "Redis health check failed ✗"
        exit 1
    fi
    
    # Check PostgreSQL
    if docker-compose exec postgres pg_isready -U recon_user > /dev/null 2>&1; then
        log_info "PostgreSQL health check passed ✓"
    else
        log_error "PostgreSQL health check failed ✗"
        exit 1
    fi
    
    log_info "All health checks passed ✓"
}

# Rollback deployment
rollback() {
    log_warn "Rolling back deployment..."
    
    # Stop current containers
    docker-compose down
    
    # Start with previous version
    VERSION="previous" docker-compose up -d
    
    log_info "Rollback completed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "check")
            check_prerequisites
            ;;
        "build")
            check_prerequisites
            build_images
            ;;
        "test")
            check_prerequisites
            run_tests
            ;;
        "deploy")
            check_prerequisites
            build_images
            run_tests
            deploy
            ;;
        "rollback")
            rollback
            ;;
        "health")
            health_check
            ;;
        *)
            echo "Usage: $0 {check|build|test|deploy|rollback|health}"
            echo ""
            echo "Commands:"
            echo "  check    - Check prerequisites"
            echo "  build    - Build Docker images"
            echo "  test     - Run test suite"
            echo "  deploy   - Full deployment (build + test + deploy)"
            echo "  rollback - Rollback to previous version"
            echo "  health   - Run health checks"
            exit 1
            ;;
    esac
}

main "$@"

---

# Makefile - Development and deployment automation
.PHONY: help install dev test build deploy clean lint format

# Variables
PROJECT_NAME := reconsideration-system
PYTHON_VERSION := 3.11
VENV_NAME := venv
DOCKER_COMPOSE := docker-compose

# Help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Setup
install: ## Install dependencies
	python -m venv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install --upgrade pip
	$(VENV_NAME)/bin/pip install -r requirements.txt
	$(VENV_NAME)/bin/pip install -r requirements-dev.txt
	$(VENV_NAME)/bin/python -m spacy download en_core_web_sm

dev: ## Start development environment
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up -d
	$(VENV_NAME)/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Testing
test: ## Run test suite
	$(VENV_NAME)/bin/pytest tests/ -v --cov=main --cov-report=html

test-integration: ## Run integration tests
	$(VENV_NAME)/bin/pytest tests/ -v -m integration

test-performance: ## Run performance tests
	$(VENV_NAME)/bin/pytest tests/ -v -m performance

test-load: ## Run load tests with Locust
	$(VENV_NAME)/bin/locust -f tests/locustfile.py --host=http://localhost:8000

# Code Quality
lint: ## Run linting
	$(VENV_NAME)/bin/flake8 main.py tests/
	$(VENV_NAME)/bin/mypy main.py
	$(VENV_NAME)/bin/bandit -r main.py

format: ## Format code
	$(VENV_NAME)/bin/black main.py tests/
	$(VENV_NAME)/bin/isort main.py tests/

# Building and Deployment
build: ## Build Docker images
	docker build -f Dockerfile.api -t $(PROJECT_NAME)-api:latest .
	docker build -f Dockerfile.worker -t $(PROJECT_NAME)-worker:latest .

deploy-local: ## Deploy locally with Docker Compose
	$(DOCKER_COMPOSE) up -d

deploy-staging: ## Deploy to staging environment
	./deploy.sh deploy staging

deploy-prod: ## Deploy to production environment
	./deploy.sh deploy production

# Monitoring and Maintenance
logs: ## View application logs
	$(DOCKER_COMPOSE) logs -f reconsideration-api

logs-worker: ## View worker logs
	$(DOCKER_COMPOSE) logs -f celery-worker

metrics: ## View metrics dashboard
	@echo "Opening Grafana dashboard: http://localhost:3000"
	@echo "Opening Prometheus: http://localhost:9090"
	@echo "Opening Flower (Celery monitoring): http://localhost:5555"

backup: ## Backup databases
	docker exec $(PROJECT_NAME)_postgres_1 pg_dump -U recon_user reconsideration_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
	docker exec $(PROJECT_NAME)_redis_1 redis-cli --rdb backup_redis_$(shell date +%Y%m%d_%H%M%S).rdb

# Cleanup
clean: ## Clean up containers and images
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

clean-all: ## Clean up everything including volumes
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# Development utilities
shell: ## Open Python shell with app context
	$(VENV_NAME)/bin/python -c "from main import *; import asyncio"

db-migrate: ## Run database migrations
	$(VENV_NAME)/bin/alembic upgrade head

db-reset: ## Reset database
	$(DOCKER_COMPOSE) exec postgres psql -U recon_user -d reconsideration_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	make db-migrate

# Monitoring and debugging
debug: ## Start application in debug mode
	$(VENV_NAME)/bin/python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn main:app --reload

monitor: ## Monitor system resources
	$(DOCKER_COMPOSE) exec reconsideration-api top

redis-cli: ## Connect to Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

psql: ## Connect to PostgreSQL
	$(DOCKER_COMPOSE) exec postgres psql -U recon_user -d reconsideration_dbp
