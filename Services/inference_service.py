from .base_service import BaseService
import logging

logger = logging.getLogger(__name__)

class InferenceService(BaseService):
    def __init__(self, broker, ai_model):
        super().__init__(broker)
        self.ai_model = ai_model

    async def start(self):
        await self.subscribe("image.submitted", self.handle_image_submitted)
        await super().start()

    async def handle_image_submitted(self, message: dict):
        payload = message.get("payload", {})
        image_path = payload.get("image_path")
        image_id = payload.get("image_id")
        
        logger.info(f"InferenceService processing image {image_id} at {image_path}")
        
        # Call fake AI Model
        prediction = self.ai_model.predict(image_path)
        
        # Publish result
        out_event = {
            "type": "inference_completed",
            "topic": "inference.completed",
            "payload": {
                "image_id": image_id,
                "image_path": image_path,
                "prediction": prediction
            }
        }
        await self.publish(out_event)
