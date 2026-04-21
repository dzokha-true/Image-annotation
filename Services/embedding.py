from .base_service import BaseService
import logging

logger = logging.getLogger(__name__)

class EmbeddingService(BaseService):
    def __init__(self, broker, embedder):
        super().__init__(broker)
        self.embedder = embedder

    async def start(self):
        await self.subscribe("annotation.stored", self.handle_annotation_stored)
        await super().start()

    async def handle_annotation_stored(self, message: dict):
        payload = message.get("payload", {})
        image_id = payload.get("image_id")
        document = payload.get("document", {})
        event_id = message.get("event_id")
        
        logger.info(f"EmbeddingService generating embedding for image {image_id}")
        
        # Call embedder
        embedding = self.embedder.generate(document)
        
        out_event = {
            "type": "embedding_created",
            "topic": "embedding.created",
            "event_id": event_id,
            "payload": {
                "image_id": image_id,
                "embedding": embedding
            }
        }
        await self.publish(out_event)
