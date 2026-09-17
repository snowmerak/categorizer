"""Local LFM prompt router used as a category scorer."""

import math
import os
from pathlib import Path
from threading import Lock

from category_scoring import CategoryInputTooLong, SiblingScore


LFM_MODEL_NAME = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"
LFM_MODEL_DIR = Path(__file__).resolve().parent / "models" / "lfm2.5-encoder-350m-prompt-router"


class LfmCategoryScorer:
    model_name = LFM_MODEL_NAME

    def __init__(self, model_dir: Path = LFM_MODEL_DIR):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not (model_dir / "model.safetensors").is_file() or not (
            model_dir / "modeling_lfm2_bidirectional.py"
        ).is_file():
            raise FileNotFoundError(
                f"LFM model not found at {model_dir}. Run `uv run python download_lfm.py` first."
            )

        requested_device = os.getenv("LFM_DEVICE", "auto")
        if requested_device == "auto":
            free_bytes = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            self.device = "cuda" if free_bytes >= 2_500_000_000 else "cpu"
        elif requested_device in ("cpu", "cuda"):
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise ValueError("LFM_DEVICE=cuda requires a CUDA device")
            self.device = requested_device
        else:
            raise ValueError("LFM_DEVICE must be auto, cpu, or cuda")

        # Upstream route() builds float32 pooling tensors, so the model must use float32.
        dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_dir, dtype=dtype, trust_remote_code=True, local_files_only=True
        ).to(self.device).eval()
        self.max_length = int(os.getenv("LFM_MAX_LENGTH", "4096"))
        if self.max_length < 1:
            raise ValueError("LFM_MAX_LENGTH must be positive")
        self.lock = Lock()

    def score(self, query: str, documents: list[str]) -> list[SiblingScore]:
        # The router returns sorted results, so include an index to map them back.
        routes = [f"{index + 1}. {' '.join(document.split())}" for index, document in enumerate(documents)]
        combined = self.model._prefix(routes) + query
        with self.lock:
            token_count = len(self.tokenizer(combined)["input_ids"])
            if token_count > self.max_length:
                raise CategoryInputTooLong(
                    f"LFM input has {token_count} tokens, exceeding LFM_MAX_LENGTH={self.max_length}"
                )
            results = self.model.route(query, routes, tokenizer=self.tokenizer)

        by_route = {item["route"]: item["score"] for item in results}
        if len(by_route) != len(routes):
            raise ValueError("The LFM model returned an unexpected number of category scores")
        return [
            SiblingScore(
                log_probability=math.log(by_route[route]) if by_route[route] > 0 else -math.inf
            )
            for route in routes
        ]
