import pytest
import asyncio
from Services.broker import MockBroker
from Services.vector_db import VectorDBService
from deps import VectorRepository

@pytest.mark.asyncio
async def test_vector_db_service():
    broker = MockBroker()
    vector_repo = VectorRepository()
    
    vector_db_svc = VectorDBService(broker, vector_repo)
    await vector_db_svc.start()
    
    published_events = []
    async def capture_event(message):
        published_events.append(message)
        
    await broker.subscribe("embedding.stored", capture_event)
    
    incoming_msg = {
        "type": "embedding_created",
        "topic": "embedding.created",
        "event_id": "test-123",
        "payload": {
            "image_id": "img1",
            "embedding": [0.5, 0.6],
            "document": {"prediction": "cat"}
        }
    }
    
    await vector_db_svc.handle_embedding_created(incoming_msg)
    await asyncio.sleep(0.01)
    
    assert len(published_events) == 1
    assert "img1" in vector_repo.saved_vectors
    assert vector_repo.saved_vectors["img1"] == [0.5, 0.6]
    assert published_events[0]["payload"]["document"] == {"prediction": "cat"}
    
    # Test idempotency
    await vector_db_svc.handle_embedding_created(incoming_msg)
    await asyncio.sleep(0.01)
    assert len(published_events) == 1 # still 1
