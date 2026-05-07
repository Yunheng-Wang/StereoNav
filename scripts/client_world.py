"""WorldModel Client

A drop-in replacement for the WorldModel that communicates via HTTP.
Uses binary tensor transmission (torch.save/load) instead of JSON for speed.
Use this in the `streamvln` environment (Python 3.9) with Habitat.
"""
import io
import requests
import torch
from typing import List


class WorldModelClient:
    """HTTP client that mimics the WorldModel interface."""

    def __init__(self, server_url: str = "http://localhost:7300"):
        self.server_url = server_url.rstrip('/')
        self._check_connection()

    def _check_connection(self):
        """Check if the server is available."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Connected to WorldModel server: {response.json()}")
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to WorldModel server at {self.server_url}: {e}")

    def eval(self):
        """Dummy eval() to match model interface."""
        pass

    def to(self, device):
        """Dummy to() to match model interface."""
        return self

    def inference(
        self,
        instruction: List[str],
        Panoramic: torch.Tensor,   # [B, 6, 3, 448, 448]
    ) -> dict:
        """Send inference request to the server via binary transmission.

        Args:
            instruction: List of instruction strings [B]
            Panoramic: Panoramic images tensor [B, 6, 3, 448, 448]

        Returns:
            Dictionary with 'responses' key containing list of generated texts
        """
        # Pack all data into a single binary payload
        buf = io.BytesIO()
        torch.save({
            'instruction': instruction,
            'panoramic': Panoramic.cpu(),
        }, buf)

        response = requests.post(
            f"{self.server_url}/inference",
            data=buf.getvalue(),
            headers={'Content-Type': 'application/octet-stream'},
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Inference failed (HTTP {response.status_code}): {response.text}")

        # Unpack binary response
        result = torch.load(io.BytesIO(response.content), weights_only=False)
        return result
