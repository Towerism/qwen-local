"""Pinned sources. ADR 0001 pins llama.cpp and ADR 0002 pins the model files.

Change these only together with the ADRs; tests compare the two.
"""

from dataclasses import dataclass

LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_TAG = "b11669"
LLAMA_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"

HF_REPO = "unsloth/Qwen3.8-27B-GGUF"
HF_REVISION = "4ca720788d1e01f1bff70c033e0d0028fd02e502"


@dataclass(frozen=True)
class PinnedFile:
    repo_path: str
    local_name: str
    size: int
    sha256: str
    vision_only: bool = False


FILES = (
    PinnedFile(
        repo_path="Qwen3.8-27B-UD-Q4_K_M.gguf",
        local_name="Qwen3.8-27B-UD-Q4_K_M.gguf",
        size=16_464_440_224,
        sha256="322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
    ),
    PinnedFile(
        repo_path="MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
        local_name="mtp-Qwen3.8-27B-Q4_0.gguf",
        size=1_369_590_656,
        sha256="50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e",
    ),
    PinnedFile(
        repo_path="mmproj-BF16.gguf",
        local_name="mmproj-BF16.gguf",
        size=931_146_432,
        sha256="83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53",
        vision_only=True,
    ),
)


def files(vision=False):
    return tuple(f for f in FILES if vision or not f.vision_only)


def download_url(f):
    return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{f.repo_path}"
