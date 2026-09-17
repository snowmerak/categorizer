"""Download and verify a fixed model revision in the Git-ignored model directory."""

import hashlib

from huggingface_hub import hf_hub_download, snapshot_download

from main import MODEL_DIR, MODEL_NAME


MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
WEIGHTS_SIZE = 1_191_588_280
WEIGHTS_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"


def verified(path, expected_size, expected_sha256):
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def download_weights(repo_id, model_dir, revision, expected_size, expected_sha256):
    target = model_dir / "model.safetensors"
    if verified(target, expected_size, expected_sha256):
        return

    hf_hub_download(
        repo_id,
        "model.safetensors",
        revision=revision,
        local_dir=model_dir,
        force_download=target.is_file(),
    )
    if not verified(target, expected_size, expected_sha256):
        raise RuntimeError("Model weights failed the size or SHA-256 check; retry the download")


def download_model(repo_id, model_dir, revision, weights_size, weights_sha256, extra_patterns=()):
    snapshot_download(
        repo_id,
        revision=revision,
        local_dir=model_dir,
        allow_patterns=["*.json", "*.model", "*.txt", *extra_patterns],
    )
    download_weights(repo_id, model_dir, revision, weights_size, weights_sha256)
    print(f"Model cached at {model_dir}")


if __name__ == "__main__":
    download_model(MODEL_NAME, MODEL_DIR, MODEL_REVISION, WEIGHTS_SIZE, WEIGHTS_SHA256)
