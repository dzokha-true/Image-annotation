import pytest
import asyncio
from Services.broker import MockBroker
from Services.inference_service import InferenceService
from Services.document_db import DocumentDBService
from Services.embedding import EmbeddingService
from Services.cli import CLI
from Services.uploader_service import UploaderService
from Services.vector_db import VectorDBService
from deps import AIModel, DocumentRepository, Embedder, VectorRepository
import os

@pytest.fixture
def test_image_path(tmpdir):
    test_image_dir = os.path.join(tmpdir, "images")
    os.makedirs(test_image_dir, exist_ok=True)
    test_img_path = os.path.join(test_image_dir, "integration_test.png")
    with open(test_img_path, "w") as f:
        f.write("dummy image content")
    return test_img_path

from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_end_to_end_pipeline(test_image_path):
    broker = MockBroker()
    
    ai_model = AIModel()
    ai_model.predict = MagicMock(return_value={"predictions": [{"label": "cat", "confidence": 0.99, "boxes": [10, 20, 30, 40]}]})
    db_repo = DocumentRepository()
    embedder = Embedder()  # Use mock embedder for fast tests
    vector_repo = VectorRepository()
    
    inferenceServ = InferenceService(broker, ai_model)
    dbServ = DocumentDBService(broker, db_repo)
    EmbedServ = EmbeddingService(broker, embedder)
    uploaderServ = UploaderService(broker)
    vectorDbServ = VectorDBService(broker, vector_repo)
    cli = CLI(broker)
    
    await asyncio.gather(
        inferenceServ.start(),
        dbServ.start(),
        EmbedServ.start(),
        uploaderServ.start(),
        vectorDbServ.start(),
        cli.start()
    )
    
    # Trigger event
    await cli.trigger_image_submission(test_image_path)
    
    # Wait for the chain to complete via the mock broker
    await asyncio.sleep(0.1)
    
    # Check that image was processed
    assert cli.unnotified_count == 1
    assert len(cli.completed_images) == 1
    
    image_id = list(cli.completed_images.keys())[0]
    
    # Check db and vectors
    assert image_id in db_repo.saved_image_ids
    assert image_id in vector_repo.saved_vectors
    assert vector_repo.saved_vectors[image_id] == [0.1, 0.2, 0.3, 0.4]
    
    # Check annotation extraction
    assert "predictions" in cli.completed_images[image_id]
    
    # Test Idempotency Integration: Trigger again with the same event
    # To truly test idempotency at a pipeline level, we can inject an event directly
    # that uses the exact same image_id.
    duplicate_event = {
        "type": "upload_requested",
        "topic": "image.upload_requested",
        "event_id": "test-123",
        "payload": {
            "image_id": image_id,
            "image_path": test_image_path
        }
    }
    
    await broker.publish("image.upload_requested", duplicate_event)
    await asyncio.sleep(0.1)
    
    # Should not process twice
    assert len(db_repo.saved_image_ids) == 1
    assert len(vector_repo.saved_vectors) == 1
    # Note: Uploader and Inference might run again because we didn't add idempotency there, 
    # but DocumentDBService handles idempotency and will return False on save(), 
    # breaking the chain before embedding is generated!
    
    assert cli.unnotified_count == 1 # still 1 because it broke at DB
