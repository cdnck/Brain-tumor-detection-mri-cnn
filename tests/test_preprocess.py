import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.preprocess import load_image, IMG_SIZE
import tempfile
import cv2


def test_load_image_shape():
    """load_image should return (224, 224, 3) normalized array."""
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, img)
        result = load_image(f.name)

    assert result.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_load_image_bad_path():
    """load_image should raise on missing file."""
    try:
        load_image("/nonexistent/path.jpg")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
