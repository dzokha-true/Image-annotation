class AIModel:
    def predict(self, image_path: str) -> dict:
        """Mock AI prediction."""
        return {"boxes": [10, 20, 30, 40], "confidence": 0.95}


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
    def generate(self, data: dict) -> list:
        """Mock embedding generation."""
        return [0.1, 0.2, 0.3, 0.4]
