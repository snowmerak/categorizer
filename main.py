"""Local choices API backed by Qwen3-Reranker-0.6B."""

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Callable

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


MODEL_DIR = Path(__file__).resolve().parent / "models" / "qwen3-reranker-0.6b"
MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a user query, determine whether the candidate response correctly answers the query."
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class ChoicesRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    choices: list[Annotated[str, Field(min_length=1, max_length=10_000)]] = Field(
        min_length=1, max_length=64
    )


class ChoiceScore(BaseModel):
    index: int
    choice: str
    yes_logit: float
    no_logit: float
    yes_probability: float
    no_probability: float
    percentage: float


class ChoicesResponse(BaseModel):
    query: str
    model: str
    choices: list[ChoiceScore]


class QwenReranker:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. Run `uv run python download_model.py` first."
            )

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, padding_side="left", local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, local_files_only=True
        ).to(self.device).eval()
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.no_token_id == self.yes_token_id or any(
            token_id == self.tokenizer.unk_token_id
            for token_id in (self.no_token_id, self.yes_token_id)
        ):
            raise ValueError("The tokenizer does not provide distinct yes/no tokens")

        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.max_length = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
        self.batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
        if self.max_length <= len(self.prefix_tokens) + len(self.suffix_tokens):
            raise ValueError("RERANKER_MAX_LENGTH is too small for the model prompt")
        if self.batch_size < 1:
            raise ValueError("RERANKER_BATCH_SIZE must be positive")
        self.lock = Lock()

    def score(self, query: str, choices: list[str]) -> list[tuple[float, float]]:
        """Return one (yes logit, no logit) pair per choice, in input order."""
        results = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(choices), self.batch_size):
                batch = choices[start : start + self.batch_size]
                pairs = [
                    f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {choice}"
                    for choice in batch
                ]
                tokenized = self.tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
                    return_attention_mask=False,
                )
                input_ids = [
                    self.prefix_tokens + ids + self.suffix_tokens
                    for ids in tokenized["input_ids"]
                ]
                inputs = self.tokenizer.pad(
                    {"input_ids": input_ids}, padding=True, return_tensors="pt"
                ).to(self.device)
                logits = self.model(
                    **inputs, logits_to_keep=1, use_cache=False
                ).logits[:, -1, :]
                yes_logits = logits[:, self.yes_token_id].float().cpu().tolist()
                no_logits = logits[:, self.no_token_id].float().cpu().tolist()
                results.extend(zip(yes_logits, no_logits))
        return results


def make_response(query: str, choices: list[str], logits: list[tuple[float, float]]) -> ChoicesResponse:
    if len(logits) != len(choices):
        raise ValueError("The model returned an unexpected number of scores")
    margins = [yes - no for yes, no in logits]
    largest = max(margins)
    weights = [math.exp(margin - largest) for margin in margins]
    total = sum(weights)
    scores = []
    for index, (choice, (yes, no), weight) in enumerate(zip(choices, logits, weights)):
        # A two-class softmax is the independent yes/no judgment for this choice.
        reference = max(yes, no)
        yes_weight = math.exp(yes - reference)
        no_weight = math.exp(no - reference)
        yes_probability = yes_weight / (yes_weight + no_weight)
        scores.append(
            ChoiceScore(
                index=index,
                choice=choice,
                yes_logit=yes,
                no_logit=no,
                yes_probability=yes_probability,
                no_probability=no_weight / (yes_weight + no_weight),
                percentage=100 * weight / total,
            )
        )
    return ChoicesResponse(query=query, model=MODEL_NAME, choices=scores)


def create_app(ranker_factory: Callable[[], QwenReranker] = QwenReranker) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ranker = ranker_factory()
        yield

    app = FastAPI(title="Qwen choices API", lifespan=lifespan)

    @app.post("/choices", response_model=ChoicesResponse)
    def rank_choices(body: ChoicesRequest, request: Request) -> ChoicesResponse:
        return make_response(
            body.query, body.choices, request.app.state.ranker.score(body.query, body.choices)
        )

    return app


app = create_app()
