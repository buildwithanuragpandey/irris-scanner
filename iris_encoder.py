import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import ssl
import os

# Bypass SSL certificate verification for model downloads (fix for macOS environment issues)
ssl._create_default_https_context = ssl._create_unverified_context

class IrisEncoder:
    """
    Generates IrisCode (binary) and CNN embeddings from normalized iris strips.
    """
    def __init__(self):
        # Gabor filter parameters
        self.orientations = [0, 45, 90, 135]  # degrees
        self.frequencies = [0.1, 0.15, 0.2]   # cycles/pixel
        self.kernel_size = (31, 31)
        
        # Load lightweight CNN for secondary features
        self.cnn_model = models.mobilenet_v2(pretrained=True)
        # Replace classifier with identity or a smaller embedding layer
        self.cnn_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.cnn_model.last_channel, 128)
        )
        self.cnn_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def gabor_filter(self, img, theta, freq):
        """Apply complex Gabor filter to image and return real/imag signs."""
        sigma = 1.0 / freq
        # OpenCV's getGaborKernel returns a real kernel. 
        # For complex, we need to shift the phase or use manual implementation.
        # We'll use 0 phase for real and pi/2 for imaginary.
        
        kernel_real = cv2.getGaborKernel(self.kernel_size, sigma, np.deg2rad(theta), 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
        kernel_imag = cv2.getGaborKernel(self.kernel_size, sigma, np.deg2rad(theta), 1.0/freq, 0.5, np.pi/2, ktype=cv2.CV_32F)
        
        real_resp = cv2.filter2D(img, cv2.CV_32F, kernel_real)
        imag_resp = cv2.filter2D(img, cv2.CV_32F, kernel_imag)
        
        return (real_resp > 0).astype(np.uint8), (imag_resp > 0).astype(np.uint8)

    def generate_iris_code(self, normalized_strip):
        """
        Generate 2048-bit IrisCode (1024 texture bits + 1024 mask bits).
        Note: Scaled down here for demonstration, but logic applies.
        """
        img = normalized_strip.astype(np.float32) / 255.0
        code_bits = []
        mask_bits = []
        
        # We apply filters and sample bits from specific locations
        # For a 64x512 strip, we can sample 32x32 = 1024 points
        rows = np.linspace(0, normalized_strip.shape[0]-1, 16, dtype=int)
        cols = np.linspace(0, normalized_strip.shape[1]-1, 32, dtype=int)
        
        for theta in self.orientations:
            for freq in self.frequencies:
                real_bin, imag_bin = self.gabor_filter(img, theta, freq)
                
                # Sample bits
                for r in rows:
                    for c in cols:
                        code_bits.append(real_bin[r, c])
                        code_bits.append(imag_bin[r, c])
                        
                        # Generate mask (0 for bad region, 1 for good)
                        # Here we mask out very dark/bright regions (eyelids/reflections)
                        pixel_val = normalized_strip[r, c]
                        mask_val = 1 if (10 < pixel_val < 245) else 0
                        mask_bits.append(mask_val)
                        mask_bits.append(mask_val)

        # Truncate or pad to exactly 1024 each for standard 2048-bit IrisCode
        # For this implementation, we take the first 1024 bits
        final_code = np.packbits(code_bits[:1024])
        final_mask = np.packbits(mask_bits[:1024])
        
        return final_code.tobytes(), final_mask.tobytes()

    def extract_cnn_features(self, normalized_strip):
        """Extract 128-d CNN embedding."""
        # Convert grayscale to RGB for MobileNet
        rgb_img = cv2.cvtColor(normalized_strip, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb_img)
        input_tensor = self.transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.cnn_model(input_tensor)
        
        return embedding.numpy().flatten()

    def encode(self, normalized_strip):
        """Full encoding pipeline."""
        iris_code, mask = self.generate_iris_code(normalized_strip)
        cnn_emb = self.extract_cnn_features(normalized_strip)
        
        return {
            "iris_code": iris_code,
            "mask": mask,
            "cnn_embedding": cnn_emb.astype(np.float32)
        }

if __name__ == "__main__":
    # Test
    encoder = IrisEncoder()
    mock_iris = np.random.randint(0, 255, (64, 512), dtype=np.uint8)
    features = encoder.encode(mock_iris)
    print(f"IrisCode Length: {len(features['iris_code'])} bytes")
    print(f"CNN Embedding Shape: {features['cnn_embedding'].shape}")
