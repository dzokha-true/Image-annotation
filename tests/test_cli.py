import pytest
import asyncio
from Services.broker import MockBroker
from Services.cli import CLI

@pytest.mark.asyncio
async def test_cli_trigger():
    broker = MockBroker()
    cli = CLI(broker)
    await cli.start()
    
    published_events = []
    async def capture_event(message):
        published_events.append(message)
        
    await broker.subscribe("image.upload_requested", capture_event)
    
    await cli.trigger_image_submission("/path/to/img.png")
    await asyncio.sleep(0.01)
    
    assert len(published_events) == 1
    assert published_events[0]["payload"]["image_path"] == "/path/to/img.png"
    assert "image_id" in published_events[0]["payload"]
    
@pytest.mark.asyncio
async def test_cli_handle_embedding_stored():
    broker = MockBroker()
    cli = CLI(broker)
    await cli.start()
    
    assert len(cli.completed_images) == 0
    assert cli.unnotified_count == 0
    
    msg = {
        "payload": {
            "image_id": "img1",
            "document": {
                "prediction": {"box": 1}
            }
        }
    }
    await cli.handle_embedding_stored(msg)
    
    assert "img1" in cli.completed_images
    assert cli.unnotified_count == 1
    assert cli.completed_images["img1"] == {"box": 1}
