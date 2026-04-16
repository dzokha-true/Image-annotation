import asyncio
import logging
from deps import AIModel, DocumentRepository, Embedder
from Services.broker import RedisBroker
from Services.inference_service import InferenceService
from Services.document_db import DocumentDBService
from Services.embedding import EmbeddingService
from Services.cli import CLI

logging.basicConfig(level=logging.INFO)

async def main():
    # Setup shared dependencies
    ai_model = AIModel()
    db_repo = DocumentRepository()
    embedder = Embedder()
    
    # Setup broker
    # In a real setup, requires Redis running locally on default port
    broker = RedisBroker("redis://localhost:6379")
    
    # Instantiate services
    inference_svc = InferenceService(broker, ai_model)
    db_svc = DocumentDBService(broker, db_repo)
    embedding_svc = EmbeddingService(broker, embedder)
    cli = CLI(broker)

    # Start all services concurrently
    tasks = [
        inference_svc.start(),
        db_svc.start(),
        embedding_svc.start()
    ]
    
    # Normally we do asyncio.gather(*tasks) and then block.
    # To demonstrate functionality inline, we also schedule a CLI submission shortly after.
    await asyncio.gather(*tasks)
    
    print("Services started, submitting a test image...")
    await cli.trigger_image_submission("/tmp/images/test_img.png")
    
    # For demonstration, keep main loop alive briefly
    await asyncio.sleep(1)
    
if __name__ == "__main__":
    asyncio.run(main())
