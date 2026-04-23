from .base_service import BaseService
import logging

logger = logging.getLogger(__name__)

class VectorDBService(BaseService):
    def __init__(self, broker, vector_repo):
        super().__init__(broker)
        self.vector_repo = vector_repo

    async def start(self):
        await self.subscribe("embedding.created", self.handle_embedding_created)
        await super().start()

    async def handle_embedding_created(self, message: dict):
        payload = message.get("payload", {})
        image_id = payload.get("image_id")
        embedding = payload.get("embedding")
        document = payload.get("document", {})
        event_id = message.get("event_id")
        
        logger.info(f"VectorDBService saving embedding for image {image_id}")
        
        saved = self.vector_repo.save(image_id, embedding)
        if not saved:
            logger.info(f"VectorDBService ignoring duplicate event for image_id={image_id}")
            return
            
        out_event = {
            "type": "embedding_stored",
            "topic": "embedding.stored",
            "event_id": event_id,
            "payload": {
                "image_id": image_id,
                "document": document
            }
        }
        await self.publish(out_event)
