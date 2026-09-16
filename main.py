"""Local choices and hierarchical categorization API backed by Qwen3-Reranker."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


MODEL_DIR = Path(__file__).resolve().parent / "models" / "qwen3-reranker-0.6b"
MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a user query, determine whether the candidate response correctly answers the query."
CATEGORY_INSTRUCTION = (
    "Given a user text, determine whether the category name and description "
    "accurately classify its main subject."
)
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


class CategorizeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    categories: dict[str, Any] = Field(min_length=1, max_length=64)


class CategoryScore(ChoiceScore):
    description: str


class CategoryLevel(BaseModel):
    depth: int
    selected: str
    candidates: list[CategoryScore]


class CategorizeResponse(BaseModel):
    query: str
    model: str
    path: list[str]
    levels: list[CategoryLevel]


@dataclass
class CategoryNode:
    name: str
    description: str
    children: list["CategoryNode"]


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

    def score(
        self, query: str, choices: list[str], instruction: str = INSTRUCTION
    ) -> list[tuple[float, float]]:
        """Return one (yes logit, no logit) pair per choice, in input order."""
        results = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(choices), self.batch_size):
                batch = choices[start : start + self.batch_size]
                pairs = [
                    f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {choice}"
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


def parse_categories(categories: dict[str, Any]) -> list[CategoryNode]:
    """Parse {name: [description, {child_name: [...]}]} into bounded nodes."""
    count = 0

    def parse_level(mapping: dict[str, Any], depth: int) -> list[CategoryNode]:
        nonlocal count
        if depth >= 16:
            raise ValueError("Category depth cannot exceed 16 levels")
        if len(mapping) > 64:
            raise ValueError("Each level can contain at most 64 categories")

        nodes = []
        for name, value in mapping.items():
            count += 1
            if count > 512:
                raise ValueError("The tree can contain at most 512 categories")
            if not isinstance(name, str) or not name.strip() or len(name) > 200:
                raise ValueError("Category names must contain 1 to 200 characters")
            if not isinstance(value, list) or not value:
                raise ValueError(f"Category {name!r} must be [description, children...]")
            description = value[0]
            if (
                not isinstance(description, str)
                or not description.strip()
                or len(description) > 10_000
            ):
                raise ValueError(f"Category {name!r} needs a nonempty description")

            children = {}
            for group in value[1:]:
                if not isinstance(group, dict) or not group:
                    raise ValueError(f"Children of {name!r} must be nonempty JSON objects")
                for child_name, child_value in group.items():
                    if child_name in children:
                        raise ValueError(f"Duplicate child category {child_name!r} in {name!r}")
                    children[child_name] = child_value
            nodes.append(
                CategoryNode(
                    name=name,
                    description=description,
                    children=parse_level(children, depth + 1) if children else [],
                )
            )
        return nodes

    return parse_level(categories, 0)


def categorize(query: str, roots: list[CategoryNode], ranker: QwenReranker) -> CategorizeResponse:
    path = []
    levels = []
    current = roots
    while current:
        documents = [
            f"Category: {' > '.join([*path, node.name])}\nDescription: {node.description}"
            for node in current
        ]
        scores = make_response(
            query,
            documents,
            ranker.score(query, documents, instruction=CATEGORY_INSTRUCTION),
        ).choices
        best_index = max(range(len(scores)), key=lambda index: scores[index].percentage)
        candidates = [
            CategoryScore(
                **score.model_dump(exclude={"choice"}),
                choice=node.name,
                description=node.description,
            )
            for node, score in zip(current, scores)
        ]
        selected = current[best_index]
        levels.append(
            CategoryLevel(depth=len(path), selected=selected.name, candidates=candidates)
        )
        path.append(selected.name)
        current = selected.children
    return CategorizeResponse(query=query, model=MODEL_NAME, path=path, levels=levels)


def create_app(ranker_factory: Callable[[], QwenReranker] = QwenReranker) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ranker = ranker_factory()
        yield

    app = FastAPI(title="Qwen categorizer API", lifespan=lifespan)

    @app.post("/choices", response_model=ChoicesResponse)
    def rank_choices(body: ChoicesRequest, request: Request) -> ChoicesResponse:
        return make_response(
            body.query, body.choices, request.app.state.ranker.score(body.query, body.choices)
        )

    @app.post("/categorize", response_model=CategorizeResponse)
    def categorize_query(body: CategorizeRequest, request: Request) -> CategorizeResponse:
        try:
            roots = parse_categories(body.categories)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return categorize(body.query, roots, request.app.state.ranker)

    return app


app = create_app()
