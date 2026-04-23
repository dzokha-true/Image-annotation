import pytest
import asyncio
from Services.broker import MockBroker
from Services.embedding import EmbeddingService
from deps import Embedder

@pytest.mark.asyncio
async def test_embedding_redis():
    broker = MockBroker()
    embedder = Embedder()
    
    embedderServ = EmbeddingService(broker, embedder)
    await embedderServ.start()
    
    published_events = []
    async def capture_event(message):
        published_events.append(message)
        
    await broker.subscribe("embedding.created", capture_event)
    
    await embedderServ.handle_annotation_stored({
        "type": "annotation_stored",
        "topic": "annotation.stored",
        "event_id": "test-123",
        "payload": {
            "image_id": "img1",
            "document": {
                "image_id": "img1",
                "image_path": "/tmp/img1.png",
                "prediction": {"boxes": [10, 20]}
            }
        }
    })
    
    await asyncio.sleep(0.01)
    
    assert len(published_events) == 1
    event = published_events[0]
    assert event["type"] == "embedding_created"
    assert event["payload"]["image_id"] == "img1"
    assert event["payload"]["embedding"] == [0.1, 0.2, 0.3, 0.4]
