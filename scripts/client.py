"""StereoVLN Model Client

A drop-in replacement for the StereoVLN model that communicates via HTTP.
Uses binary tensor transmission (torch.save/load) instead of JSON for speed.
Use this in the evaluation environment (Python 3.9) with Habitat.

Typical usage:
    client = StereoVLNClient(server_url="http://localhost:7200")
    outputs = client.inference(instruction=[...], ...)
"""
import io
import requests
import torch
from typing import List


class StereoVLNClient:
    """HTTP client that mimics the StereoVLN model interface."""

    def __init__(self, server_url: str = "http://localhost:5080"):
        self.server_url = server_url.rstrip('/')
        self._check_connection()

    def _check_connection(self):
        """Check if the server is available."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Connected to StereoVLN server: {response.json()}")
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to StereoVLN server at {self.server_url}: {e}")

    def eval(self):
        """Dummy eval() to match model interface."""
        pass

    def to(self, device):
        """Dummy to() to match model interface."""
        return self

    def inference(
        self,
        instruction: List[str],
        history_action: List[str],
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448]
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448]
        left_history_video: torch.Tensor,   # [B, history_num, 3, 448, 448]
        right_history_video: torch.Tensor,  # [B, history_num, 3, 448, 448]
        max_new_tokens: int = 24,
        depth_iters: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        output_point: bool = False,
        output_depth: bool = False,
    ) -> dict:
        """Send inference request to the server via binary transmission."""
        # Pack all data into a single binary payload
        buf = io.BytesIO()
        torch.save({
            'instruction': instruction,
            'history_action': history_action,
            'left_current_frame': left_current_frame.cpu(),
            'right_current_frame': right_current_frame.cpu(),
            'left_history_video': left_history_video.cpu(),
            'right_history_video': right_history_video.cpu(),
            'max_new_tokens': max_new_tokens,
            'depth_iters': depth_iters,
            'temperature': temperature,
            'top_p': top_p,
            'output_point': output_point,
            'output_depth': output_depth,
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
