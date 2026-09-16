# Choices API

[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)를 로컬에서 실행하는 FastAPI 서비스입니다. Windows와 NVIDIA GPU(CUDA 12.8 PyTorch 빌드)를 기준으로 설정했습니다.

```powershell
uv sync
uv run python download_model.py
uv run python download_embedding.py
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

모델은 `models/qwen3-reranker-0.6b/`에 저장되며 Git에 포함되지 않습니다. 서버는 시작할 때 이 로컬 파일만 읽으므로, 다운로드 후에는 Hugging Face 연결 없이 추론합니다. API 문서는 `http://127.0.0.1:8000/docs`에서 볼 수 있습니다.

## 요청

`POST /choices`

```json
{
  "query": "대한민국의 수도는 어디야?",
  "choices": [
    "대한민국의 수도는 서울입니다.",
    "대한민국의 수도는 부산입니다.",
    "화성은 붉은 행성입니다."
  ]
}
```

PowerShell 예시:

```powershell
$body = @{
  query = "대한민국의 수도는 어디야?"
  choices = @("대한민국의 수도는 서울입니다.", "대한민국의 수도는 부산입니다.")
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/choices -Method Post -ContentType application/json -Body $body
```

각 선택지는 입력 순서대로 `index`, `choice`, `yes_logit`, `no_logit`, `yes_probability`, `no_probability`, `percentage`와 함께 반환됩니다. `yes_probability`와 `no_probability`는 해당 선택지에 대한 두 토큰의 softmax 결과입니다. `percentage`는 각 선택지의 `yes_logit - no_logit`에 다시 softmax를 적용한 상대 적합도이며, 한 응답의 합계는 100%입니다. 이 모델은 응답의 관련성을 비교하는 리랭커이므로 사실 검증기로 간주할 수 없고, 이 값은 객관적으로 보정된 정답 확률도 아닙니다.

한 요청에서 선택지는 1~64개까지 받습니다. 기본값은 추론 배치 크기 4, 입력 길이 1024토큰입니다. 필요하면 서버 시작 전에 `RERANKER_BATCH_SIZE`, `RERANKER_MAX_LENGTH` 환경 변수를 설정할 수 있습니다. 긴 입력은 모델 토큰화 과정에서 잘립니다. GPU가 없는 환경에서는 CPU로 실행되며 속도가 느릴 수 있습니다.

## 계층형 분류

`POST /categorize`는 `query`와 카테고리 트리 `categories`를 받습니다. 트리는 `{ "이름": ["설명", { "자식 이름": ["설명", ...] }] }` 형태이며, 자식이 없는 노드는 `["설명"]`으로 끝납니다.

```json
{
  "query": "우리 집 고양이가 사료를 안 먹어요",
  "top_k": 3,
  "categories": {
    "동물": [
      "동물에 관한 이야기",
      {
        "포유류": [
          "젖을 먹여 새끼를 기르는 동물",
          {"고양이": ["반려묘에 관한 이야기"], "개": ["반려견에 관한 이야기"]}
        ],
        "조류": ["새에 관한 이야기"]
      }
    ],
    "기계": ["기계와 장치에 관한 이야기"],
    "분류불가": ["어느 분류에도 속하지 않는 이야기"]
  }
}
```

각 단계에서 형제 카테고리의 **전체 경로와 설명**을 배치로 채점합니다. 각 후보의 경로 비율은 `부모 경로 비율 × 해당 단계의 percentage / 100`으로 계산합니다. 높은 비율의 노드부터 자식을 확장하여 상위 말단 경로를 찾습니다. 예를 들어 A가 60%여도 자식이 50%라면 경로는 30%이고, B가 40%에 자식이 100%라면 B 경로가 앞섭니다.

`top_k`는 반환할 말단 경로 개수(1~10, 기본값 1)입니다. 응답의 `ranked_paths`에는 경로와 곱셈 결과인 `percentage`가 높은 순서대로 담깁니다. 상위 일부만 반환하므로 이 비율들의 합이 꼭 100%일 필요는 없습니다. 기존 `path`와 `levels`는 1위 경로를 나타냅니다. `levels`의 후보 비율은 해당 단계의 형제 사이에서 합계 100%입니다. `분류불가`를 말단 선택지로 트리에 넣으면 다른 경로와 함께 순위에 포함됩니다. `top_k`가 커질수록 더 많은 가지를 채점하므로 응답 시간이 늘어날 수 있습니다.

한 단계에는 최대 64개, 트리 전체에는 최대 512개 카테고리를 넣을 수 있고 최대 깊이는 16단계입니다. 이름은 1~200자, 설명은 1~10,000자입니다. 잘못된 트리 구조는 HTTP 422로 거절합니다.

```powershell
uv run python -m unittest discover -s tests
```

## 넓은 단계의 임베딩 후보 축소

기본 설정에서는 한 단계의 형제가 32개 이상일 때만 임베딩으로 상위 8개를 고른 뒤 리랭커로 채점합니다. 31개 이하라면 임베딩 모델을 로드하지 않고 모든 후보를 바로 채점합니다. 임베딩 모델은 다음 명령으로 로컬에 캐시합니다.

```powershell
uv run python download_embedding.py
```

임베딩 모델은 `models/qwen3-embedding-0.6b/`에 저장되며 Git에 포함되지 않습니다. `prefilter_min_siblings`(기본 32)와 `prefilter_top_n`(기본 8)은 요청마다 바꿀 수 있습니다. `prefilter_top_n: null`이면 모든 후보를 채점합니다. 기본적으로 이름이 `분류불가` 또는 `Unclassified`인 후보는 축소 대상에서 제외하며, `prefilter_keep_names`로 보호할 이름을 지정할 수 있습니다.

```json
{
  "query": "고양이 사료를 찾고 있어",
  "top_k": 3,
  "prefilter_top_n": 2,
  "prefilter_min_siblings": 5,
  "prefilter_keep_names": ["분류불가"],
  "categories": {
    "반려동물": ["반려동물 관련 요청"],
    "식품": ["식품 관련 요청"],
    "의류": ["옷과 패션 관련 요청"],
    "교통": ["이동 수단 관련 요청"],
    "분류불가": ["어느 분류에도 해당하지 않는 요청"]
  }
}
```

위 예시는 임계값을 5개로 낮췄으므로 형제 5개 중 임베딩 상위 2개를 리랭커에 보내고, `분류불가`는 순위와 무관하게 추가로 남깁니다. 따라서 리랭커에 전달되는 수는 `prefilter_top_n`을 넘을 수도 있습니다. 응답의 `search_mode`는 실제 축소가 일어났을 때 `prefiltered`, 그렇지 않으면 `exact`입니다. `omitted_candidates`는 탐색 중 방문한 단계에서 제외된 후보 수입니다. `prefiltered`일 때 `levels`와 `ranked_paths`의 비율은 **남긴 후보 안에서** 계산하므로 전체 트리의 정확한 순위나 확률로 해석하면 안 됩니다. 임베딩이 적합한 후보를 놓치면 결과 경로에서도 빠질 수 있습니다.

첫 요청은 임베딩 모델 로드와 문서 임베딩 생성 때문에 느릴 수 있습니다. 같은 이름, 경로, 설명의 후보 문서는 메모리에 캐시되어 후속 요청에 재사용됩니다. `EMBEDDING_DEVICE`는 `auto`(기본), `cuda`, `cpu` 중 하나이며, `auto`는 리랭커 로드 후 CUDA 여유 메모리가 2.5GB 이상이면 GPU를 사용합니다. 필요한 경우 `EMBEDDING_BATCH_SIZE`(기본 8), `EMBEDDING_MAX_LENGTH`(기본 512)를 설정할 수 있습니다.
