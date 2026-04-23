import os
import shutil
import uuid
import logging
from .base_service import BaseService

logger = logging.getLogger(__name__)

class UploaderService(BaseService):
    async def start(self):
        await self.subscribe("image.upload_requested", self.handle_upload_requested)
        await super().start()

    async def handle_upload_requested(self, message: dict):
        payload = message.get("payload", {})
        local_path = payload.get("image_path")
        image_id = payload.get("image_id", str(uuid.uuid4()))
        event_id = message.get("event_id")
        
        logger.info(f"UploaderService copying image from {local_path}")
        
        if not os.path.exists("/tmp"):
            os.makedirs("/tmp", exist_ok=True)
            
        ext = os.path.splitext(local_path)[1]
        if not ext:
            ext = ".png" # default fallback
        
        dest_path = f"/tmp/{image_id}{ext}"
        
        try:
            shutil.copy2(local_path, dest_path)
        except Exception as e:
            logger.error(f"UploaderService failed to copy file {local_path} to {dest_path}: {e}")
            return
            
        # Publish result
        out_event = {
            "type": "image_submitted",
            "topic": "image.submitted",
            "event_id": event_id,
            "payload": {
                "image_id": image_id,
                "image_path": dest_path
            }
        }
        await self.publish(out_event)
