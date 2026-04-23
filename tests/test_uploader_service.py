import pytest
import asyncio
import os
from Services.broker import MockBroker
from Services.uploader_service import UploaderService

@pytest.mark.asyncio
async def test_uploader_service(tmpdir):
    broker = MockBroker()
    uploader_svc = UploaderService(broker)
    await uploader_svc.start()
    
    published_events = []
    async def capture_event(message):
        published_events.append(message)
        
    await broker.subscribe("image.submitted", capture_event)
    
    # Create fake image
    img_path = os.path.join(tmpdir, "test.png")
    with open(img_path, "w") as f:
        f.write("fake content")
        
    incoming_msg = {
        "type": "upload_requested",
        "topic": "image.upload_requested",
        "event_id": "event-1",
        "payload": {
            "image_id": "img1",
            "image_path": img_path
        }
    }
    
    await uploader_svc.handle_upload_requested(incoming_msg)
    await asyncio.sleep(0.01)
    
    assert len(published_events) == 1
    payload = published_events[0]["payload"]
    assert payload["image_id"] == "img1"
    assert payload["image_path"] == "/tmp/img1.png"
    assert os.path.exists("/tmp/img1.png")
