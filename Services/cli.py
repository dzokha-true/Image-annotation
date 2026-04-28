from .base_service import BaseService
import logging
import uuid
import asyncio
import readline

logger = logging.getLogger(__name__)

class CLI(BaseService):
    def __init__(self, broker, embedder=None, vector_repo=None):
        super().__init__(broker)
        self.completed_images = {}
        self.unnotified_count = 0
        self.embedder = embedder
        self.vector_repo = vector_repo
        
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
                print("  search <query>: Semantic search using natural language (e.g. 'a cat').")
                print("  status        : Check how many images have been fully processed.")
                print("  help          : Show this help message.")
                print("  exit          : Shut down the CLI.")
            elif command.lower() == 'exit':
                print("Exiting CLI.")
                break
            elif command.lower().startswith("upload "):
                parts = command.split(" ", 1)
                if len(parts) > 1:
                    path = parts[1].strip("'\" ")
                    await self.trigger_image_submission(path)
                    print(f"Upload requested for {path}.")
                else:
                    print("Please provide a path: upload <path>")
            elif command.lower().startswith("search "):
                parts = command.split(" ", 1)
                if len(parts) > 1:
                    query = parts[1].strip()
                    if not self.embedder or not self.vector_repo:
                        print("Search is not available. Missing embedder or vector DB.")
                    else:
                        print(f"Searching for '{query}'...")
                        # Run the synchronous inference and search in a separate thread
                        text_embedding = await asyncio.to_thread(self.embedder.generate_text, query)
                        if not text_embedding:
                            print("Failed to generate embedding for the query.")
                        else:
                            results = await asyncio.to_thread(self.vector_repo.search, text_embedding, 3)
                            if not results:
                                print("No matches found.")
                            else:
                                print(f"\nTop matches for '{query}':")
                                for res in results:
                                    img_id = res['image_id']
                                    dist = res['distance']
                                    print(f"- Image ID: {img_id} (Distance: {dist:.4f})")
                                    if img_id in self.completed_images:
                                        print(f"  Annotations: {self.completed_images[img_id]}")
                else:
                    print("Please provide a query: search <query>")
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
