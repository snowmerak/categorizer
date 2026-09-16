"""Download the optional local embedding model used for candidate prefiltering."""

from download_model import download_model
from embedding import EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME


EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
WEIGHTS_SIZE = 1_191_586_416
WEIGHTS_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"


if __name__ == "__main__":
    download_model(
        EMBEDDING_MODEL_NAME,
        EMBEDDING_MODEL_DIR,
        EMBEDDING_REVISION,
        WEIGHTS_SIZE,
        WEIGHTS_SHA256,
    )
