"""Download and verify a fixed model revision in the Git-ignored model directory."""

import hashlib

from huggingface_hub import hf_hub_url, snapshot_download
import requests

from main import MODEL_DIR, MODEL_NAME


MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
WEIGHTS_SIZE = 1_191_588_280
WEIGHTS_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"


def verified(path):
    if not path.is_file() or path.stat().st_size != WEIGHTS_SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == WEIGHTS_SHA256


def download_weights():
    target = MODEL_DIR / "model.safetensors"
    if verified(target):
        return

    partial = MODEL_DIR / "model.safetensors.part"
    if verified(partial):
        partial.replace(target)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= WEIGHTS_SIZE:
        partial.unlink()
        offset = 0

    url = hf_hub_url(MODEL_NAME, "model.safetensors", revision=MODEL_REVISION)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0  # The server ignored Range; restart instead of appending duplicate bytes.
        downloaded = offset
        next_report = ((offset // 100_000_000) + 1) * 100_000_000
        with partial.open("ab" if offset else "wb") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"Downloaded {downloaded / WEIGHTS_SIZE:.0%} of model weights", flush=True)
                        next_report += 100_000_000

    if not verified(partial):
        raise RuntimeError("Model weights failed the size or SHA-256 check; retry the download")
    partial.replace(target)


if __name__ == "__main__":
    snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.model", "*.txt"],
    )
    download_weights()
    print(f"Model cached at {MODEL_DIR}")
