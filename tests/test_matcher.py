import pytest
import numpy as np
from iris_matcher import IrisMatcher
from config import settings

def test_hamming_distance_identical():
    matcher = IrisMatcher()
    code = b'\xAA' * 128
    mask = b'\xFF' * 128
    hd = matcher.compute_hamming_distance(code, mask, code, mask)
    assert hd == 0.0

def test_hamming_distance_different():
    matcher = IrisMatcher()
    code_a = b'\x00' * 128
    code_b = b'\xFF' * 128
    mask = b'\xFF' * 128
    hd = matcher.compute_hamming_distance(code_a, mask, code_b, mask)
    assert hd == 1.0

def test_rotational_compensation():
    matcher = IrisMatcher()
    code_a = b'\xAA' * 128
    mask = b'\xFF' * 128
    # Shifted version
    code_b = np.roll(np.frombuffer(code_a, dtype=np.uint8), 1).tobytes()
    hd = matcher.compute_hamming_distance(code_a, mask, code_b, mask)
    # Rotating should maintain low HD
    assert hd < 0.1

def test_cosine_similarity():
    matcher = IrisMatcher()
    emb_a = np.array([1.0, 0.0], dtype=np.float32)
    emb_b = np.array([1.0, 0.0], dtype=np.float32)
    sim = matcher.compute_cosine_similarity(emb_a, emb_b)
    assert pytest.approx(sim) == 1.0
    
    emb_c = np.array([0.0, 1.0], dtype=np.float32)
    sim_diff = matcher.compute_cosine_similarity(emb_a, emb_c)
    assert pytest.approx(sim_diff) == 0.0

def test_match_logic():
    matcher = IrisMatcher()
    f1 = {
        "iris_code": b'\xAA' * 128,
        "mask": b'\xFF' * 128,
        "cnn_embedding": np.ones(128, dtype=np.float32)
    }
    f2 = {
        "iris_code": b'\xAA' * 128,
        "mask": b'\xFF' * 128,
        "cnn_embedding": np.ones(128, dtype=np.float32)
    }
    result = matcher.match(f1, f2)
    assert result["matched"] is True
    assert result["confidence"] > 0.9
