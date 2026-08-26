import numpy as np
from config import settings

class IrisMatcher:
    """
    Implements a two-stage matching pipeline: Hamming Distance + CNN Cosine Similarity.
    """
    def compute_hamming_distance(self, code_a, mask_a, code_b, mask_b):
        """
        Compute Hamming distance between two IrisCodes with bitwise masking.
        Handles rotational compensation by shifting code_a.
        """
        # Convert bytes to numpy arrays for bitwise ops
        arr_a = np.frombuffer(code_a, dtype=np.uint8)
        arr_mask_a = np.frombuffer(mask_a, dtype=np.uint8)
        arr_b = np.frombuffer(code_b, dtype=np.uint8)
        arr_mask_b = np.frombuffer(mask_b, dtype=np.uint8)

        def get_hd(code1, code2, m1, m2):
            diff = np.bitwise_xor(code1, code2)
            combined_mask = np.bitwise_and(m1, m2)
            
            # Count set bits in (diff AND mask) and compare to set bits in mask
            pop_diff = bin(int.from_bytes(np.bitwise_and(diff, combined_mask).tobytes(), 'big')).count('1')
            pop_mask = bin(int.from_bytes(combined_mask.tobytes(), 'big')).count('1')
            
            if pop_mask == 0: return 1.0
            return pop_diff / pop_mask

        # Rotational compensation: shift arr_a by small amounts
        # In a real iris code (cylindrical), shifting corresponds to eye tilt
        # We'll try shifts of -8 to +8 bits
        min_hd = 1.0
        for shift in range(-8, 9):
            # Simple byte-level shift for demonstration
            shifted_a = np.roll(arr_a, shift % len(arr_a))
            shifted_mask_a = np.roll(arr_mask_a, shift % len(arr_mask_a))
            hd = get_hd(shifted_a, arr_b, shifted_mask_a, arr_mask_b)
            if hd < min_hd:
                min_hd = hd
        
        return min_hd

    def compute_cosine_similarity(self, emb_a, emb_b):
        """Compute cosine similarity between 128-d CNN vectors."""
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0: return 0.0
        
        similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)
        return similarity

    def match(self, features_a, features_b):
        """
        Two-stage matching logic.
        Stage 1: Fast Hamming Filter
        Stage 2: Precision CNN Similarity
        """
        hd = self.compute_hamming_distance(
            features_a["iris_code"], features_a["mask"],
            features_b["iris_code"], features_b["mask"]
        )
        
        # Stage 1: Threshold
        if hd >= settings.HAMMING_THRESHOLD:
            return {
                "matched": False,
                "confidence": 1.0 - hd,
                "hamming_distance": hd,
                "cosine_similarity": 0.0,
                "stage": 1
            }
            
        # Stage 2: CNN verification
        similarity = self.compute_cosine_similarity(
            features_a["cnn_embedding"], features_b["cnn_embedding"]
        )
        
        is_match = similarity > settings.COSINE_SIM_THRESHOLD and hd < settings.HAMMING_THRESHOLD
        
        # Combined confidence score
        confidence = (1.0 - hd) * 0.5 + similarity * 0.5
        
        return {
            "matched": bool(is_match),
            "confidence": float(confidence),
            "hamming_distance": float(hd),
            "cosine_similarity": float(similarity),
            "stage": 2
        }

if __name__ == "__main__":
    # Test
    matcher = IrisMatcher()
    code1 = b'\xAA' * 128
    mask1 = b'\xFF' * 128
    # Slightly different code
    code2 = b'\xAB' * 128
    mask2 = b'\xFF' * 128
    
    emb1 = np.random.rand(128).astype(np.float32)
    emb2 = emb1 + np.random.normal(0, 0.01, 128).astype(np.float32)
    
    result = matcher.match(
        {"iris_code": code1, "mask": mask1, "cnn_embedding": emb1},
        {"iris_code": code2, "mask": mask2, "cnn_embedding": emb2}
    )
    print(f"Match Result: {result}")
