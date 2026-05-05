# Barys-Eyes: Event-Driven Image Annotation Pipeline

Barys-Eyes is a modular, event-driven image annotation and semantic search system built with Python, `asyncio`, and Redis Pub-Sub. The architecture is fully decoupled, with microservices communicating exclusively through events, allowing for a highly scalable and robust image processing pipeline.

## Features

- **Event-Driven Architecture**: Fully asynchronous system leveraging Redis Pub/Sub for inter-service communication.
- **Hardware-Accelerated ML Models**: Uses PyTorch with support for MPS (Apple Silicon), CUDA, or CPU.
- **Automated Object Detection**: Integrates Hugging Face DETR (`facebook/detr-resnet-50`) to generate automatic object bounding boxes and labels.
- **Semantic Search**: Utilizes OpenAI CLIP (`openai/clip-vit-base-patch32`) to generate image and text embeddings, coupled with FAISS for fast and accurate natural language similarity searches.
- **Interactive CLI**: A built-in command-line interface for uploading images and querying the database with natural language.

## Architecture

The system consists of several independent microservices:

- **`InferenceService`**: Listens for new images and performs object detection using the DETR model.
- **`EmbeddingService`**: Generates CLIP embeddings for newly uploaded images.
- **`VectorDBService`**: Manages the FAISS vector database to store embeddings and handle semantic search queries.
- **`DocumentDBService`**: Acts as the primary database, ensuring idempotency and storing the final metadata and annotations for each processed image.
- **`UploaderService`**: Handles the ingestion of new image files and triggers the pipeline.
- **`CLI`**: The interactive frontend that allows users to upload files and perform semantic searches.

## Requirements

- Python 3.9+
- A running Redis server (local or remote)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The first time you run the models, it will download the DETR and CLIP weights from Hugging Face.)*

## Configuration

The system uses a `.env` file for configuration. Create a `.env` file in the root directory:

```env
# Optional: Set your Redis broker URL. Defaults to redis://localhost:6379
REDIS_URL=redis://localhost:6379
```

## Usage

1. Start your Redis server:
   ```bash
   redis-server
   ```
   *(Or using Docker: `docker run -p 6379:6379 redis`)*

2. Start the pipeline and interactive CLI:
   ```bash
   python main.py
   ```

3. From the CLI, you can use commands such as:
   - `upload <path_to_image>`: Ingest an image into the pipeline.
   - `search <text_query>`: Perform a semantic search across processed images.

## Testing

The project includes a comprehensive test suite using `pytest` and `pytest-asyncio`. To run the tests:

```bash
pytest
```
