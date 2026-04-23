from .base_service import BaseService
import logging
import uuid
import asyncio

logger = logging.getLogger(__name__)

class CLI(BaseService):
    def __init__(self, broker):
        super().__init__(broker)
        self.completed_images = {}
        self.unnotified_count = 0
        
    async def start(self):
        await self.subscribe("embedding.stored", self.handle_embedding_stored)
        await super().start()
        
    async def handle_embedding_stored(self, message: dict):
        payload = message.get("payload", {})
        image_id = payload.get("image_id")
        document = payload.get("document", {})
        prediction = document.get("prediction", {})
        
        if image_id:
            self.completed_images[image_id] = prediction
            self.unnotified_count += 1

    async def trigger_image_submission(self, image_path: str):
        """Manually trigger a payload, publishing image.upload_requested."""
        img_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())
        # logger.info(f"CLI trigger: Submitting image {image_path} with id {img_id}")
        
        out_event = {
            "type": "upload_requested",
            "topic": "image.upload_requested",
            "event_id": event_id,
            "payload": {
                "image_id": img_id,
                "image_path": image_path
            }
        }
        await self.publish(out_event)
        
    async def interactive_loop(self):
        print("\n" + "="*50)
        print("Welcome to Barys-Eyes Image Annotation Pipeline!")
        print("The system is fully initialized and ready.")
        print("Type 'help' to see the list of available commands.")
        print("="*50)
        
        while True:
            if self.unnotified_count > 0:
                print(f"[CLI Notification] {self.unnotified_count} new uploads have been fully processed.")
                self.unnotified_count = 0
                
            # Using to_thread for non-blocking input
            command = await asyncio.to_thread(input, "\nBarys-Eyes > ")
            command = command.strip()
            
            if not command:
                continue
                
            if command.lower() == 'help':
                print("Available commands:")
                print("  upload <path> : Submit an image to the pipeline.")
                print("  status        : Check how many images have been fully processed.")
                print("  help          : Show this help message.")
                print("  exit          : Shut down the CLI.")
            elif command.lower() == 'exit':
                print("Exiting CLI.")
                break
            elif command.lower().startswith("upload "):
                parts = command.split(" ", 1)
                if len(parts) > 1:
                    path = parts[1]
                    await self.trigger_image_submission(path)
                    print(f"Upload requested for {path}.")
                else:
                    print("Please provide a path: upload <path>")
            elif command.lower() == 'status':
                if not self.completed_images:
                    print("No images have completed the pipeline yet.")
                else:
                    ans = await asyncio.to_thread(input, f"{len(self.completed_images)} images have finished the pipeline. Do you want to see the details? (y/n): ")
                    if ans.lower() == 'y':
                        print("\nCompleted Uploads & Annotations:")
                        for img_id, pred in self.completed_images.items():
                            print(f"- ID: {img_id}")
                            print(f"  Annotations: {pred}")
            else:
                print("Unknown command. Type 'help' for a list of available commands.")
