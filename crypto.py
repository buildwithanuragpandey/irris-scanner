import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class CryptoHelper:
    """
    Handles encryption/decryption of biometric data using AES-256-GCM.
    In production, base_key should be fetched from a secure Vault/HSM.
    """
    def __init__(self, key: bytes = None):
        # 256-bit key for AES-256
        self.key = key or os.urandom(32)
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypts data and returns nonce + ciphertext + tag."""
        nonce = os.urandom(12)
        # AESGCM.encrypt(nonce, data, associated_data)
        ciphertext = self.aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypts data using the nonce at the beginning."""
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None)

# Global helper with a mock key (in reality, this would be managed per key_id)
crypto_helper = CryptoHelper(b'\x01' * 32)
