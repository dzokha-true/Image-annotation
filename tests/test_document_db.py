import pytest
import asyncio
from Services.broker import MockBroker
from Services.document_db import DocumentDBService
from deps import DocumentRepository

@pytest.mark.asyncio
async def idemtopency_test():
    broker = MockBroker()
    db_repo = DocumentRepository()
    
    db_conn = DocumentDBService(broker, db_repo)
    await db_conn.start()
    
    published_events = []
    async def capture_event(message):
        published_events.append(message)
        
    await broker.subscribe("annotation.stored", capture_event)
    
    incoming_msg = {
        "type": "inference_completed",
        "topic": "inference.completed",
        "event_id": "test-123",
        "payload": {
            "image_id": "img1",
            "image_path": "/path/to/img1.png",
            "prediction": {"boxes": [10, 20]}
        }
    }
    
    # First publication - should be saved
    await db_conn.handle_inference_completed(incoming_msg)
    await asyncio.sleep(0.01)
    
    assert len(published_events) == 1
    assert "img1" in db_repo.saved_image_ids
    
    # Second publication with same image_id - should be ignored (idempotency)
    await db_conn.handle_inference_completed(incoming_msg)
    await asyncio.sleep(0.01)
    
    # We shouldn't receive a second annotation.stored event
    assert len(published_events) == 1
