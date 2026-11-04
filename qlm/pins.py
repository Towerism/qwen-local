"""Pinned sources. ADR 0001 pins llama.cpp, and ADRs 0002 and 0005 pin the model files.

Change these only together with the ADRs; tests compare the two.
"""

from dataclasses import dataclass

LLAMA_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_TAG = "b11669"
LLAMA_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"


@dataclass(frozen=True)
class PinnedFile:
    repo: str
    revision: str
    repo_path: str
    local_name: str
    size: int
    sha256: str
    role: str  # "main", "mtp" (the multi-token prediction draft) or "mmproj" (the vision projector)


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    files: tuple


class UnknownModel(ValueError):
    pass


MODELS = (
    Model(
        id="qwen3.6-35b-a3b",
        name="Qwen3.6-35B-A3B",
        files=(
            PinnedFile(
                repo="mradermacher/Qwen3.6-35B-A3B-GGUF",
                revision="ac96549d2e8b902f1a772083a33d13b5ee01fe3c",
                repo_path="Qwen3.6-35B-A3B.Q4_K_M.gguf",
                local_name="Qwen3.6-35B-A3B.Q4_K_M.gguf",
                size=21_166_758_368,
                sha256="1f05a92890b796172f69171ac8d84d1334f1453034d712757bea4b8184a96bac",
                role="main",
            ),
            PinnedFile(
                repo="ggml-org/Qwen3.6-35B-A3B-GGUF",
                revision="7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5",
                repo_path="mtp-Qwen3.6-35B-A3B-Q4_0.gguf",
                local_name="mtp-Qwen3.6-35B-A3B-Q4_0.gguf",
                size=1_060_038_432,
                sha256="606fca331adcbfbdadc107512ce6a7161e84e1646ba0e0018256426f6296877f",
                role="mtp",
            ),
        ),
    ),
    Model(
        id="qwen3.8-27b",
        name="Qwen3.8-27B",
        files=(
            PinnedFile(
                repo="unsloth/Qwen3.8-27B-GGUF",
                revision="4ca720788d1e01f1bff70c033e0d0028fd02e502",
                repo_path="Qwen3.8-27B-UD-Q4_K_M.gguf",
                local_name="Qwen3.8-27B-UD-Q4_K_M.gguf",
                size=16_464_440_224,
                sha256="322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
                role="main",
            ),
            PinnedFile(
                repo="unsloth/Qwen3.8-27B-GGUF",
                revision="4ca720788d1e01f1bff70c033e0d0028fd02e502",
                repo_path="MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                local_name="mtp-Qwen3.8-27B-Q4_0.gguf",
                size=1_369_590_656,
                sha256="50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e",
                role="mtp",
            ),
            PinnedFile(
                repo="unsloth/Qwen3.8-27B-GGUF",
                revision="4ca720788d1e01f1bff70c033e0d0028fd02e502",
                repo_path="mmproj-BF16.gguf",
                local_name="mmproj-BF16.gguf",
                size=931_146_432,
                sha256="83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53",
                role="mmproj",
            ),
        ),
    ),
)
DEFAULT_MODEL = "qwen3.6-35b-a3b"


def model_ids():
    return tuple(m.id for m in MODELS)


def model(model_id):
    """The catalog entry for this id. Raises UnknownModel naming the valid ids."""
    for m in MODELS:
        if m.id == model_id:
            return m
    raise UnknownModel(f"unknown model {model_id!r}. Choose one of: {', '.join(model_ids())}")


def files(model, vision=False):
    """The model's files; the vision projector only when asked for."""
    return tuple(f for f in model.files if vision or f.role != "mmproj")


def download_url(f):
    return f"https://huggingface.co/{f.repo}/resolve/{f.revision}/{f.repo_path}"
