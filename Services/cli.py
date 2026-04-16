from .base_service import BaseService
import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

class CLI(BaseService):
    def __init__(self, broker):
        super().__init__(broker)

    async def trigger_image_submission(self, image_path: str):
        """Manually trigger a payload, publishing image.submitted."""
        img_id = str(uuid.uuid4())
        logger.info(f"CLI trigger: Submitting image {image_path} with id {img_id}")
        
        out_event = {
            "type": "image_submitted",
            "topic": "image.submitted",
            "payload": {
                "image_id": img_id,
                "image_path": image_path
            }
        }
        await self.publish(out_event)
