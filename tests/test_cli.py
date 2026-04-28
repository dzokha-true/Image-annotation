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

@pytest.mark.asyncio
async def test_cli_interactive_loop():
    broker = MockBroker()
    
    class MockEmbedder:
        def generate_text(self, text):
            if text == "fail":
                return []
            return [0.1, 0.2]
            
    class MockVectorRepo:
        def search(self, emb, k):
            if emb == []:
                return []
            return [{"image_id": "img1", "distance": 0.5}]
            
    cli = CLI(broker, embedder=MockEmbedder(), vector_repo=MockVectorRepo())
    cli.unnotified_count = 1
    cli.completed_images = {"img1": "pred"}
    
    inputs = [
        "",
        "help",
        "upload /path.jpg",
        "upload",
        "search cat",
        "search fail",
        "search",
        "status",
        "y",
        "unknown",
        "exit"
    ]
    
    input_iter = iter(inputs)
    
    # We patch asyncio.to_thread to intercept input, embedder, and vector_repo calls
    original_to_thread = asyncio.to_thread
    async def mock_to_thread(func, *args, **kwargs):
        if func.__name__ == "input":
            try:
                return next(input_iter)
            except StopIteration:
                return "exit"
        return await original_to_thread(func, *args, **kwargs)
        
    import unittest.mock
    with unittest.mock.patch("asyncio.to_thread", new=mock_to_thread):
        await cli.interactive_loop()
        
    # If it breaks out of the loop without raising exceptions, the commands were processed.
    assert cli.unnotified_count == 0
