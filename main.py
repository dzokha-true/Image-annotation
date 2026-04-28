import asyncio
import logging
import os
from dotenv import load_dotenv
from deps import AIModel, DocumentRepository, CLIPEmbedder, FAISSVectorRepository
from Services.broker import RedisBroker
from Services.inference_service import InferenceService
from Services.document_db import DocumentDBService
from Services.embedding import EmbeddingService
from Services.cli import CLI
from Services.uploader_service import UploaderService
from Services.vector_db import VectorDBService

import warnings
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.WARNING)

async def main():
    # Setup shared dependencies
    ai_model = AIModel()
    db_repo = DocumentRepository()
    embedder = CLIPEmbedder()
    vector_repo = FAISSVectorRepository(dimension=512)
    
    # Setup broker
    # In a real setup, requires Redis running locally on default port
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    broker = RedisBroker(redis_url)
    
    # Instantiate services
    inference_svc = InferenceService(broker, ai_model)
    db_svc = DocumentDBService(broker, db_repo)
    embedding_svc = EmbeddingService(broker, embedder)
    uploader_svc = UploaderService(broker)
    vector_db_svc = VectorDBService(broker, vector_repo)
    cli = CLI(broker, embedder, vector_repo)

    # Start services (which subscribes them to topics)
    await inference_svc.start()
    await db_svc.start()
    await embedding_svc.start()
    await uploader_svc.start()
    await vector_db_svc.start()
    await cli.start()

    # Start the broker listening loop in the background
    asyncio.create_task(broker.start_listening())
    
    print("Services started.")
    
    # Start the interactive CLI loop
    await cli.interactive_loop()
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down.")
