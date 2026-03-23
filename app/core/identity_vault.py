import os
import json
import logging
from typing import Dict, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

class IdentityVault:
    """
    JARVIS V4 Secure Identity Vault
    Uses AES-256-GCM to securely store and retrieve social/developer API keys.
    """
    def __init__(self, key_hex: str = None):
        # AES-256 requires a 32-byte key
        key_env = os.getenv("JARVIS_VAULT_KEY")
        if key_env:
            self.key = bytes.fromhex(key_env)
        elif key_hex:
            self.key = bytes.fromhex(key_hex)
        else:
            # Generate a temporary ephemeral key if none provided
            logger.warning("No JARVIS_VAULT_KEY found. Generating ephemeral key (data will be lost on restart).")
            self.key = AESGCM.generate_key(bit_length=256)
        
        self.aesgcm = AESGCM(self.key)
        self.vault_file = "jarvis_secure_vault.enc"
        
        # In-memory decrypted cache (securely held in RAM)
        self._cache: Dict[str, str] = {}
        self._load_vault()

    def _load_vault(self):
        if os.path.exists(self.vault_file):
            try:
                with open(self.vault_file, "rb") as f:
                    ciphertext = f.read()
                # Extracted 12-byte nonce
                nonce = ciphertext[:12]
                encrypted_data = ciphertext[12:]
                decrypted = self.aesgcm.decrypt(nonce, encrypted_data, None)
                self._cache = json.loads(decrypted.decode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to decrypt vault (incorrect key or corrupted): {e}")
                self._cache = {}

    def _save_vault(self):
        try:
            plaintext = json.dumps(self._cache).encode("utf-8")
            nonce = os.urandom(12)
            ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
            with open(self.vault_file, "wb") as f:
                f.write(nonce + ciphertext)
        except Exception as e:
            logger.error(f"Failed to write vault: {e}")

    def store_credential(self, platform: str, token: str):
        """Securely store an OAuth token or API key for a specific platform."""
        self._cache[platform] = token
        self._save_vault()
        logger.info(f"🔒 Securely vaulted credential for [{platform}] using AES-256")

    def get_credential(self, platform: str) -> Optional[str]:
        """Retrieve a credential for authorized agents."""
        return self._cache.get(platform)

# Global singleton
identity_vault = IdentityVault()
