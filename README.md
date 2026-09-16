# Choices API

[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)를 로컬에서 실행하는 FastAPI 서비스입니다. Windows와 NVIDIA GPU(CUDA 12.8 PyTorch 빌드)를 기준으로 설정했습니다.

```powershell
uv sync
uv run python download_model.py
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
