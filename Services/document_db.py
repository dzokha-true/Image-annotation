from .base_service import BaseService
import logging

logger = logging.getLogger(__name__)

class DocumentDBService(BaseService):
    def __init__(self, broker, db_conn):
        super().__init__(broker)
        self.db_conn = db_conn

    async def start(self):
        await self.subscribe("inference.completed", self.handle_inference_completed)
        await super().start()

    async def handle_inference_completed(self, message: dict):
        payload = message.get("payload", {})
        image_id = payload.get("image_id")
        event_id = message.get("event_id")
        
        logger.info(f"DocumentDBService saving annotations for image {image_id}")
        
        document = {
            "image_id": image_id,
            "image_path": payload.get("image_path"),
            "prediction": payload.get("prediction")
        }
        
        # Checking for idempotency
        saved = self.db_conn.save(document) # Returns false if already saved
        if not saved:
            logger.info(f"DocumentDBService ignoring duplicate event for image_id={image_id}")
            return
            
        out_event = {
            "type": "annotation_stored",
            "topic": "annotation.stored",
            "event_id": event_id,
            "payload": {
                "image_id": image_id,
                "document": document
            }
        }
        await self.publish(out_event)
