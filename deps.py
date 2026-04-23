import logging
logger = logging.getLogger(__name__)

class AIModel:
    def __init__(self):
        try:
            import torch
            from transformers import DetrImageProcessor, DetrForObjectDetection
            logger.info("Loading DETR Object Detection model (this may take a moment)...")
            
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
            logger.info(f"Using device: {self.device} for DETR model.")
            
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=False)
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        except ImportError:
            logger.warning("transformers/torch package not installed, AIModel (DETR) will use fallback mock.")
            self.model = None
            self.processor = None

    def predict(self, image_path: str) -> dict:
        """Real object detection using Hugging Face DETR."""
        if not self.model or not self.processor:
            return {"boxes": [10, 20, 30, 40], "confidence": 0.95} # Fallback mock

        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Let's format the results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            
            predictions = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                predictions.append({
                    "label": self.model.config.id2label[label.item()],
                    "confidence": round(score.item(), 3),
                    "boxes": box
                })
                
            return {"predictions": predictions}
        except Exception as e:
            logger.error(f"Failed to generate DETR prediction for {image_path}: {e}")
            return {"error": str(e)}


class DocumentRepository:
    def __init__(self):
        self.saved_image_ids = set()

    def save(self, document: dict) -> bool:
        """Mock database save."""
        image_id = document.get("image_id")
        if image_id in self.saved_image_ids:
            return False  # Indicate it's already saved (idempotency check)
        self.saved_image_ids.add(image_id)
        return True

class Embedder:
    def generate(self, image_path: str) -> list:
        """Mock embedding generation."""
        return [0.1, 0.2, 0.3, 0.4]

class VectorRepository:
    def __init__(self):
        self.saved_vectors = {}

    def save(self, image_id: str, embedding: list) -> bool:
        """Mock vector database save."""
        if image_id in self.saved_vectors:
            return False
        self.saved_vectors[image_id] = embedding
        return True


class CLIPEmbedder:
    def __init__(self):
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            logger.info("Loading CLIP model (this may take a moment)...")
            
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
            logger.info(f"Using device: {self.device} for CLIP model.")
                
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        except ImportError:
            logger.warning("transformers/torch package not installed, CLIPEmbedder will not function properly.")
            self.model = None
            self.processor = None

    def generate(self, image_path: str) -> list:
        """Real embedding generation using OpenAI CLIP."""
        if not self.model or not self.processor:
            return [0.1, 0.2, 0.3, 0.4] # Fallback mock

        try:
            from PIL import Image
            import torch
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            # Move back to CPU before converting to list, if it was on MPS/CUDA
            return outputs.cpu().squeeze().tolist()
        except Exception as e:
            logger.error(f"Failed to generate CLIP embedding for {image_path}: {e}")
            return []

