import pytest
import asyncio
from Services.broker import MockBroker
from Services.inference_service import InferenceService
from deps import AIModel

from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_inference_redis():
    broker = MockBroker()
    ai_model = AIModel()
    ai_model.predict = MagicMock(return_value={"predictions": [{"label": "cat", "confidence": 0.99, "boxes": [10, 20, 30, 40]}]})
    
    InferenceServ = InferenceService(broker, ai_model)
    await InferenceServ.start()
    
    # We will subscribe to what it should publish
    published_events = []
    async def capture_event(message):
        published_events.append(message)
        
    await broker.subscribe("inference.completed", capture_event)
    
    await InferenceServ.handle_image_submitted({
        "type": "image_submitted",
        "topic": "image.submitted",
        "event_id": "test-123",
        "payload": {
            "image_id": "img1",
            "image_path": "/path/to/img1.png"
        }
    })
    
    # Since MockBroker publishes immediately using asyncio.create_task,
    # yield control back to the event loop momentarily to let callbacks execute.
    await asyncio.sleep(0.01)
    
    assert len(published_events) == 1
    event = published_events[0]
    assert event["type"] == "inference_completed"
    assert event["payload"]["image_id"] == "img1"
    assert "prediction" in event["payload"]
    assert "predictions" in event["payload"]["prediction"]
    assert event["payload"]["prediction"]["predictions"][0]["boxes"] == [10, 20, 30, 40]
