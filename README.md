# Choices API

A FastAPI service that runs [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) locally. The environment is configured for Windows and NVIDIA GPUs with a CUDA 12.8 PyTorch build.

```powershell
uv sync
uv run python download_model.py
uv run python download_embedding.py
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

The models are cached under `models/` and excluded from Git. After downloading them, inference reads local files without contacting Hugging Face. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Rank choices

`POST /choices` accepts a user query and 1 to 64 candidate responses:

```json
{
  "query": "What is the capital of South Korea?",
  "choices": [
    "The capital of South Korea is Seoul.",
    "The capital of South Korea is Busan.",
    "Mars is known as the red planet."
  ]
}
```

PowerShell example:

```powershell
$body = @{
  query = "What is the capital of South Korea?"
  choices = @("The capital of South Korea is Seoul.", "The capital of South Korea is Busan.")
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/choices -Method Post -ContentType application/json -Body $body
```

Each choice is returned in input order with `index`, `choice`, `yes_logit`, `no_logit`, `yes_probability`, `no_probability`, and `percentage`. The yes/no probabilities are a two-token softmax for that choice. `percentage` applies another softmax across the choices' `yes_logit - no_logit` margins, so the percentages sum to 100%. These are relative reranker scores, not calibrated probabilities of factual correctness.

The default inference batch size is 4 and the maximum input length is 1,024 tokens. Set `RERANKER_BATCH_SIZE` or `RERANKER_MAX_LENGTH` before starting the server to change them. Long inputs are truncated during tokenization. Inference also works on a CPU, but may be slower.

## Hierarchical categorization

`POST /categorize` accepts a `query` and a category tree in `categories`. Each node uses the shape `{ "name": ["description", { "child name": ["description", ...] }] }`. A leaf ends with `["description"]`.

```json
{
  "query": "My cat won't eat its food",
  "top_k": 3,
  "categories": {
    "Animals": [
      "Topics about animals",
      {
        "Mammals": [
          "Animals that nurse their young",
          {"Cats": ["Topics about pet cats"], "Dogs": ["Topics about pet dogs"]}
        ],
        "Birds": ["Topics about birds"]
      }
    ],
    "Machines": ["Topics about machines and devices"],
    "Unclassified": ["Topics that fit none of these categories"]
  }
}
```

At each level, the reranker scores sibling categories in a batch using each candidate's **full path and description**. A child path's percentage is its parent's path percentage multiplied by that child's percentage at the current level, divided by 100. The search expands high-scoring nodes first to find the highest-ranked leaf paths. For example, a 60% parent with a 50% child yields a 30% path; a 40% parent with a 100% child yields a 40% path and ranks higher.

`top_k` controls how many leaf paths are returned (1 to 10; default 1). `ranked_paths` lists those paths and their multiplied percentages in descending order. Because only the top paths are returned, their percentages need not sum to 100%. `path` and `levels` describe the highest-ranked path. Candidate percentages within each level sum to 100% across the siblings scored at that level. An `Unclassified` leaf can compete with other paths when included in the tree. Larger `top_k` values may require scoring more branches and increase response time.

Each level can have up to 64 categories, the whole tree up to 512 categories, and the depth up to 16 levels. Names must contain 1 to 200 characters and descriptions 1 to 10,000 characters. Invalid trees receive HTTP 422.

## Embedding prefilter for wide levels

By default, a level with **32 or more siblings** uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) to select the top 8 candidates before reranking them. Levels with 31 or fewer siblings go straight to the reranker, so they do not load the embedding model. Cache that model locally with `uv run python download_embedding.py`; it is stored in `models/qwen3-embedding-0.6b/` and excluded from Git.

You can change `prefilter_min_siblings` (default 32) and `prefilter_top_n` (default 8) per request. Set `prefilter_top_n` to `null` to rerank every candidate. Candidates named `분류불가` or `Unclassified` are kept by default; `prefilter_keep_names` replaces that list with the names you provide.

This example lowers the threshold to demonstrate prefiltering with five siblings:

```json
{
  "query": "I'm looking for cat food",
  "top_k": 3,
  "prefilter_top_n": 2,
  "prefilter_min_siblings": 5,
  "prefilter_keep_names": ["Unclassified"],
  "categories": {
    "Pet supplies": ["Requests related to pets"],
    "Food": ["Requests related to food"],
    "Clothing": ["Requests related to clothes and fashion"],
    "Transportation": ["Requests related to travel and vehicles"],
    "Unclassified": ["Requests that fit none of these categories"]
  }
}
```

The reranker receives the embedding's top two candidates plus `Unclassified` if it was not already selected. The number of reranked candidates can therefore exceed `prefilter_top_n`. The response has `search_mode: "prefiltered"` when candidates were omitted, or `search_mode: "exact"` otherwise. `omitted_candidates` counts candidates excluded at levels visited during the search. In prefiltered mode, percentages in `levels` and `ranked_paths` are calculated **only among retained candidates**. A relevant path can be missed if the embedding step excludes it.

The first prefiltered request may be slower because it loads the embedding model and encodes category documents. Documents with the same path and description are cached in memory for later requests. `EMBEDDING_DEVICE` can be `auto` (default), `cuda`, or `cpu`. In auto mode, the GPU is used when at least 2.5 GB of CUDA memory remains after loading the reranker. You can also set `EMBEDDING_BATCH_SIZE` (default 8) and `EMBEDDING_MAX_LENGTH` (default 512).

## Tests

```powershell
uv run python -m unittest discover -s tests
```
