import pytest
import sys
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_deps_aimodel_missing_torch():
    with patch.dict(sys.modules, {'torch': None, 'transformers': None}):
        from deps import AIModel
        model = AIModel()
        assert model.model is None
        
        # Test fallback mock
        res = model.predict("test.jpg")
        assert "boxes" in res

@pytest.mark.asyncio
async def test_deps_clip_missing_torch():
    with patch.dict(sys.modules, {'torch': None, 'transformers': None}):
        from deps import CLIPEmbedder
        embedder = CLIPEmbedder()
        assert embedder.model is None
        
        # Test fallback
        res = embedder.generate("test.jpg")
        assert res == [0.1, 0.2, 0.3, 0.4]
        
        text_res = embedder.generate_text("cat")
        assert text_res == [0.1, 0.2, 0.3, 0.4]

@pytest.mark.asyncio
async def test_deps_faiss_missing():
    with patch.dict(sys.modules, {'faiss': None}):
        from deps import FAISSVectorRepository
        repo = FAISSVectorRepository(dimension=512)
        assert repo.faiss_available is False
        
        # Should fallback to dict
        repo.save("img1", [0.1])
        assert "img1" in repo.saved_vectors
        assert repo.save("img1", [0.1]) is False
        
        # Search should return []
        assert repo.search([0.1]) == []

@pytest.mark.asyncio
async def test_faiss_real():
    # Only test if faiss is available
    try:
        import faiss
        import numpy as np
    except ImportError:
        pytest.skip("FAISS not installed")
        
    from deps import FAISSVectorRepository
    repo = FAISSVectorRepository(dimension=2)
    assert repo.faiss_available is True
    
    # Save
    res1 = repo.save("img1", [1.0, 0.0])
    assert res1 is True
    
    # Duplicate save
    res2 = repo.save("img1", [1.0, 0.0])
    assert res2 is False
    
    # Save another
    repo.save("img2", [0.0, 1.0])
    
    # Search
    results = repo.search([0.9, 0.1], top_k=2)
    assert len(results) > 0
    assert results[0]['image_id'] == "img1"
