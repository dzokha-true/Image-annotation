import asyncio
import logging
import os
from dotenv import load_dotenv
from deps import AIModel, DocumentRepository, CLIPEmbedder, VectorRepository
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
    vector_repo = VectorRepository()
    
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
    cli = CLI(broker)

    # Start all services concurrently in the background
    tasks = [
        asyncio.create_task(inference_svc.start()),
        asyncio.create_task(db_svc.start()),
        asyncio.create_task(embedding_svc.start()),
        asyncio.create_task(uploader_svc.start()),
        asyncio.create_task(vector_db_svc.start()),
        asyncio.create_task(cli.start())
    ]
    
    print("Services started.")
    
    # Start the interactive CLI loop
    await cli.interactive_loop()
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down.")
