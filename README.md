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

```powershell
uv run python -m unittest discover -s tests
```
