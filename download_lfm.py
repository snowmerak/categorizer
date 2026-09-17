"""Download the pinned LFM prompt router and its required custom model code."""

from download_model import download_model
from lfm import LFM_MODEL_DIR, LFM_MODEL_NAME


LFM_REVISION = "35ca4a0469f180f1cf05a630df8842fa17ac18e3"
WEIGHTS_SIZE = 1_420_051_928
WEIGHTS_SHA256 = "9fab23eeb312d951bca8a0dfa4068ca4be4d55283a66c6fe5cbe9cf1e14e631d"


if __name__ == "__main__":
    download_model(
        LFM_MODEL_NAME,
        LFM_MODEL_DIR,
        LFM_REVISION,
        WEIGHTS_SIZE,
        WEIGHTS_SHA256,
        extra_patterns=("*.py", "LICENSE"),
    )
