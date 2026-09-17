"""Local choices and hierarchical categorization API with selectable scorers."""

from dataclasses import dataclass
import heapq
from itertools import count
import math
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from category_scoring import CategoryInputTooLong, CategoryScorer, SiblingScore


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
    model: Literal["qwen", "lfm"] = "qwen"
    top_k: int = Field(default=1, ge=1, le=10)
    prefilter_top_n: int | None = Field(default=8, ge=1, le=64)
    prefilter_min_siblings: int = Field(default=32, ge=2, le=64)
    prefilter_keep_names: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=lambda: ["분류불가", "Unclassified"], max_length=16
    )


class CategoryScore(BaseModel):
    index: int
    choice: str
    description: str
    percentage: float
    yes_logit: float | None = None
    no_logit: float | None = None
    yes_probability: float | None = None
    no_probability: float | None = None


class CategoryLevel(BaseModel):
    depth: int
    selected: str
    candidates: list[CategoryScore]


class RankedPath(BaseModel):
    path: list[str]
    percentage: float


class CategorizeResponse(BaseModel):
    query: str
    model: str
    path: list[str]
    levels: list[CategoryLevel]
    ranked_paths: list[RankedPath]
    search_mode: Literal["exact", "prefiltered"]
    omitted_candidates: int


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


def log_softmax_margins(logits: list[tuple[float, float]]) -> list[float]:
    margins = [yes - no for yes, no in logits]
    largest = max(margins)
    log_total = largest + math.log(sum(math.exp(margin - largest) for margin in margins))
    return [margin - log_total for margin in margins]


def make_response(query: str, choices: list[str], logits: list[tuple[float, float]]) -> ChoicesResponse:
    if len(logits) != len(choices):
        raise ValueError("The model returned an unexpected number of scores")
    log_percentages = log_softmax_margins(logits)
    scores = []
    for index, (choice, (yes, no), log_percentage) in enumerate(
        zip(choices, logits, log_percentages)
    ):
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
                percentage=100 * math.exp(log_percentage),
            )
        )
    return ChoicesResponse(query=query, model=MODEL_NAME, choices=scores)


class QwenCategoryScorer:
    model_name = MODEL_NAME

    def __init__(self, ranker: QwenReranker):
        self.ranker = ranker

    def score(self, query: str, documents: list[str]) -> list[SiblingScore]:
        logits = self.ranker.score(query, documents, instruction=CATEGORY_INSTRUCTION)
        choices = make_response(query, documents, logits).choices
        return [
            SiblingScore(
                log_probability=log_probability,
                details=choice.model_dump(
                    include={"yes_logit", "no_logit", "yes_probability", "no_probability"}
                ),
            )
            for choice, log_probability in zip(choices, log_softmax_margins(logits))
        ]


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


def categorize(
    query: str,
    roots: list[CategoryNode],
    scorer: CategoryScorer,
    top_k: int = 1,
    prefilter_top_n: int | None = None,
    prefilter_min_siblings: int = 32,
    prefilter_keep_names: set[str] | None = None,
    embedder_factory: Callable[[], Any] | None = None,
) -> CategorizeResponse:
    """Return top leaves by sibling softmax; prefiltering makes this approximate."""
    frontier = []
    sequence = count()
    scored_levels = {}
    query_vector = None
    omitted_candidates = 0

    def add_children(path: tuple[str, ...], children: list[CategoryNode], parent_log: float):
        nonlocal query_vector, omitted_candidates
        documents = [
            f"Category: {' > '.join([*path, node.name])}\nDescription: {node.description}"
            for node in children
        ]
        original_indices = list(range(len(children)))
        if (
            prefilter_top_n is not None
            and len(children) >= prefilter_min_siblings
            and len(children) > prefilter_top_n
        ):
            if embedder_factory is None:
                raise ValueError("An embedder is required when prefiltering is enabled")
            embedder = embedder_factory()
            if query_vector is None:
                query_vector = embedder.encode_query(query)
            selected_indices = embedder.select(
                query_vector,
                documents,
                [node.name for node in children],
                prefilter_top_n,
                prefilter_keep_names or set(),
            )
            omitted_candidates += len(children) - len(selected_indices)
            original_indices = selected_indices
            children = [children[index] for index in selected_indices]
            documents = [documents[index] for index in selected_indices]
        scores = scorer.score(query, documents)
        if len(scores) != len(children):
            raise ValueError("The category scorer returned an unexpected number of scores")
        scored_levels[path] = [
            CategoryScore(
                index=original_index,
                choice=node.name,
                description=node.description,
                percentage=100 * math.exp(score.log_probability),
                **score.details,
            )
            for original_index, node, score in zip(original_indices, children, scores)
        ]
        for node, score in zip(children, scores):
            child_path = (*path, node.name)
            heapq.heappush(
                frontier,
                (-(parent_log + score.log_probability), next(sequence), node, child_path),
            )

    add_children((), roots, 0.0)
    ranked_paths = []
    # Every descendant has at most its ancestor's mass, so popped leaves are in rank order.
    while frontier and len(ranked_paths) < top_k:
        negative_log, _, node, path = heapq.heappop(frontier)
        log_mass = -negative_log
        if node.children:
            add_children(path, node.children, log_mass)
        else:
            ranked_paths.append(
                RankedPath(path=list(path), percentage=100 * math.exp(log_mass))
            )

    best_path = ranked_paths[0].path
    levels = [
        CategoryLevel(
            depth=depth,
            selected=name,
            candidates=scored_levels[tuple(best_path[:depth])],
        )
        for depth, name in enumerate(best_path)
    ]
    return CategorizeResponse(
        query=query,
        model=scorer.model_name,
        path=best_path,
        levels=levels,
        ranked_paths=ranked_paths,
        search_mode="prefiltered" if omitted_candidates else "exact",
        omitted_candidates=omitted_candidates,
    )


def create_app(
    ranker_factory: Callable[[], QwenReranker] = QwenReranker,
    embedder_factory: Callable[[], Any] | None = None,
    lfm_factory: Callable[[], CategoryScorer] | None = None,
) -> FastAPI:
    app = FastAPI(title="Categorizer API")
    app.state.ranker = None
    app.state.ranker_lock = Lock()
    app.state.embedder = None
    app.state.embedder_lock = Lock()
    app.state.lfm = None
    app.state.lfm_lock = Lock()

    def get_ranker():
        with app.state.ranker_lock:
            if app.state.ranker is None:
                try:
                    app.state.ranker = ranker_factory()
                except FileNotFoundError as error:
                    raise HTTPException(status_code=503, detail=str(error)) from error
            return app.state.ranker

    def get_embedder():
        with app.state.embedder_lock:
            if app.state.embedder is None:
                from embedding import QwenEmbedder

                factory = embedder_factory or QwenEmbedder
                try:
                    app.state.embedder = factory()
                except FileNotFoundError as error:
                    raise HTTPException(status_code=503, detail=str(error)) from error
            return app.state.embedder

    def get_lfm():
        with app.state.lfm_lock:
            if app.state.lfm is None:
                from lfm import LfmCategoryScorer

                factory = lfm_factory or LfmCategoryScorer
                try:
                    app.state.lfm = factory()
                except FileNotFoundError as error:
                    raise HTTPException(status_code=503, detail=str(error)) from error
            return app.state.lfm

    @app.post("/choices", response_model=ChoicesResponse)
    def rank_choices(body: ChoicesRequest) -> ChoicesResponse:
        return make_response(body.query, body.choices, get_ranker().score(body.query, body.choices))

    @app.post("/categorize", response_model=CategorizeResponse, response_model_exclude_none=True)
    def categorize_query(body: CategorizeRequest) -> CategorizeResponse:
        try:
            roots = parse_categories(body.categories)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if body.model == "lfm":
            prefilter_fields = {
                "prefilter_top_n", "prefilter_min_siblings", "prefilter_keep_names"
            }
            if prefilter_fields & body.model_fields_set:
                raise HTTPException(
                    status_code=422,
                    detail="Prefilter options are available only with model='qwen'",
                )
            scorer = get_lfm()
            prefilter_top_n = None
        else:
            scorer = QwenCategoryScorer(get_ranker())
            prefilter_top_n = body.prefilter_top_n
        try:
            return categorize(
                body.query,
                roots,
                scorer,
                top_k=body.top_k,
                prefilter_top_n=prefilter_top_n,
                prefilter_min_siblings=body.prefilter_min_siblings,
                prefilter_keep_names=set(body.prefilter_keep_names),
                embedder_factory=get_embedder,
            )
        except CategoryInputTooLong as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()
