"""Optional Qwen embedding prefilter for wide category levels."""

from collections import OrderedDict
import os
from pathlib import Path
from threading import Lock


EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_DIR = Path(__file__).resolve().parent / "models" / "qwen3-embedding-0.6b"
QUERY_INSTRUCTION = (
    "Given a user text, retrieve category descriptions that accurately classify its main subject"
)


class QwenEmbedder:
    def __init__(self, model_dir: Path = EMBEDDING_MODEL_DIR):
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"Embedding model not found at {model_dir}. "
                "Run `uv run python download_embedding.py` first."
            )

        requested_device = os.getenv("EMBEDDING_DEVICE", "auto")
        if requested_device == "auto":
            free_bytes = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            self.device = "cuda" if free_bytes >= 2_500_000_000 else "cpu"
        elif requested_device in ("cpu", "cuda"):
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise ValueError("EMBEDDING_DEVICE=cuda requires a CUDA device")
            self.device = requested_device
        else:
            raise ValueError("EMBEDDING_DEVICE must be auto, cpu, or cuda")

        self.torch = torch
        self.functional = functional
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, padding_side="left", local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_dir, dtype=dtype, local_files_only=True
        ).to(self.device).eval()
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
        self.max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
        if self.batch_size < 1 or self.max_length < 1:
            raise ValueError("Embedding batch size and max length must be positive")
        self.lock = Lock()
        self.document_cache = OrderedDict()
        self.max_cached_documents = 4096

    def _encode(self, texts: list[str]):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**batch, use_cache=False)
            pooled = outputs.last_hidden_state[:, -1, :].float()
            vectors.extend(self.functional.normalize(pooled, p=2, dim=1).cpu())
        return vectors

    def encode_query(self, query: str):
        instructed = f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"
        with self.lock, self.torch.inference_mode():
            return self._encode([instructed])[0]

    def select(
        self,
        query_vector,
        documents: list[str],
        names: list[str],
        top_n: int,
        keep_names: set[str],
    ) -> list[int]:
        """Return shortlisted indices in input order, preserving named choices."""
        with self.lock, self.torch.inference_mode():
            missing = list(dict.fromkeys(doc for doc in documents if doc not in self.document_cache))
            vectors = {doc: self.document_cache[doc] for doc in documents if doc in self.document_cache}
            for document, vector in zip(missing, self._encode(missing)):
                vectors[document] = vector
                self.document_cache[document] = vector
            for document in documents:
                self.document_cache.move_to_end(document)
            while len(self.document_cache) > self.max_cached_documents:
                self.document_cache.popitem(last=False)

            similarities = (self.torch.stack([vectors[doc] for doc in documents]) @ query_vector).tolist()
            pinned = {index for index, name in enumerate(names) if name in keep_names}
            ranked = sorted(
                range(len(documents)),
                key=lambda index: similarities[index],
                reverse=True,
            )
            return sorted(pinned.union(ranked[:top_n]))
