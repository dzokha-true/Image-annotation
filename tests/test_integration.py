import pytest
import asyncio
from Services.broker import MockBroker
from Services.inference_service import InferenceService
from Services.document_db import DocumentDBService
from Services.embedding import EmbeddingService
from Services.cli import CLI
from deps import AIModel, DocumentRepository, Embedder

@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    broker = MockBroker()
    
    ai_model = AIModel()
    db_repo = DocumentRepository()
    embedder = Embedder()
    
    inference_svc = InferenceService(broker, ai_model)
    db_svc = DocumentDBService(broker, db_repo)
    embedding_svc = EmbeddingService(broker, embedder)
    cli = CLI(broker)
    
    await asyncio.gather(
        inference_svc.start(),
        db_svc.start(),
        embedding_svc.start()
    )
    
    # Track final output
    final_events = []
    async def capture_embedding(message):
        final_events.append(message)
        
    await broker.subscribe("embedding.created", capture_embedding)
    
    # Trigger event
    await cli.trigger_image_submission("/tmp/images/integration_test.png", image_id="img_integration")
    
    # Wait for the chain to complete via the mock broker
    await asyncio.sleep(0.05)
    
    assert len(final_events) == 1
    assert final_events[0]["payload"]["image_id"] == "img_integration"
    assert "img_integration" in db_repo.saved_image_ids
