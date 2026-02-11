import pytest
import numpy as np
from src.preprocessing.image_utils import resize_with_pad, normalize_image

def test_resize_with_pad_dimensions():
    dummy_img = np.zeros((375, 1242, 3), dtype=np.uint8)
    resized, ratio, pad = resize_with_pad(dummy_img, target_size=(640, 640))
    assert resized.shape == (640, 640, 3)

def test_normalization_range():
    white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    normalized = normalize_image(white_img)
    assert np.max(normalized) == 1.0
    assert np.min(normalized) == 1.0