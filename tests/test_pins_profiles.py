import pathlib
import re

import pytest

from qlm import pins, profiles
from tests.conftest import REPO


ADRS = {
    "qwen3.8-27b": "docs/adr/0002-model-file-and-vram-budget.md",
    "qwen3.6-35b-a3b": "docs/adr/0005-qwen3.6-35b-a3b-model-files-and-vram-budget.md",
}
PIN_ROW = re.compile(
    r"^\| `([^`]+)` \| `([0-9a-f]{40})` \| `([^`]+)`(?: \([^)]*\))? \| ([\d,]+) \| `([0-9a-f]{64})` \|$", re.M
)


def test_llama_cpp_pin_matches_adr_0001(repo):
    adr1 = (repo / "docs/adr/0001-runtime-llama-cpp-server.md").read_text()
    assert f"tag `{pins.LLAMA_TAG}`, commit `{pins.LLAMA_COMMIT}`" in adr1


@pytest.mark.parametrize("model_id", sorted(ADRS))
def test_pins_match_the_model_adrs(repo, model_id):
    adr = (repo / ADRS[model_id]).read_text()
    rows = {(m[1], m[2], m[3], int(m[4].replace(",", "")), m[5]) for m in PIN_ROW.finditer(adr)}
    expected = {(f.repo, f.revision, f.repo_path, f.size, f.sha256) for f in pins.model(model_id).files}
    assert rows == expected


def test_every_model_has_an_adr_and_a_unique_id():
    ids = [m.id for m in pins.MODELS]
    assert sorted(ids) == sorted(ADRS)
    assert len(set(ids)) == len(ids)
    assert pins.DEFAULT_MODEL in ids


def test_every_model_has_exactly_one_main_file_and_distinct_local_names():
    for m in pins.MODELS:
        assert [f.role for f in m.files].count("main") == 1, m.id
        assert {f.role for f in m.files} <= {"main", "mtp", "mmproj"}, m.id
    names = [f.local_name for m in pins.MODELS for f in m.files]
    assert len(set(names)) == len(names)


def test_the_35b_draft_comes_from_its_own_repo_and_revision():
    main, mtp = pins.files(pins.model("qwen3.6-35b-a3b"))
    assert pins.download_url(main) == (
        "https://huggingface.co/mradermacher/Qwen3.6-35B-A3B-GGUF/resolve/"
        "ac96549d2e8b902f1a772083a33d13b5ee01fe3c/Qwen3.6-35B-A3B.Q4_K_M.gguf"
    )
    assert pins.download_url(mtp).startswith(
        "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5/"
    )


def test_an_unknown_model_id_names_the_valid_ones():
    with pytest.raises(pins.UnknownModel, match="qwen3.6-35b-a3b, qwen3.8-27b"):
        pins.model("bogus")


def test_only_the_projector_is_vision_only():
    qwen27 = pins.model("qwen3.8-27b")
    assert [f.local_name for f in pins.files(qwen27)] == ["Qwen3.8-27B-UD-Q4_K_M.gguf", "mtp-Qwen3.8-27B-Q4_0.gguf"]
    assert [f.role for f in pins.files(qwen27, vision=True)] == ["main", "mtp", "mmproj"]
    qwen35 = pins.model("qwen3.6-35b-a3b")
    assert pins.files(qwen35, vision=True) == pins.files(qwen35)
    assert [f.role for f in pins.files(qwen35)] == ["main", "mtp"]


@pytest.mark.parametrize("name, vram", [("agent", 20.3), ("long", 20.0), ("vision", 18.7)])
def test_profiles_carry_their_vram_need(repo, name, vram):
    profile = profiles.load(repo / "profiles/qwen3.8-27b" / f"{name}.env")
    assert profile.vram_gib == vram


def test_rendering_replaces_home_marker(repo, tmp_path):
    profile = profiles.load(repo / "profiles/qwen3.8-27b/agent.env")
    text = profile.render(tmp_path / "some home")
    assert "%h" not in text
    assert f"LLAMA_ARG_MODEL={tmp_path}/some home/.local/share/qlm/models/Qwen3.8-27B-UD-Q4_K_M.gguf" in text
    assert text.splitlines()[0] == "# qlm: vram_gib=20.3"


def test_vision_profile_loads_the_projector_and_drops_the_no_projector_line(repo):
    keys = profiles.load(repo / "profiles/qwen3.8-27b/vision.env").settings
    assert "LLAMA_ARG_MMPROJ" in keys
    assert "LLAMA_ARG_MMPROJ_AUTO" not in keys
    assert not any(k.startswith("LLAMA_ARG_SPEC_") for k in keys)


def test_long_profile_has_no_mtp_and_96k_context(repo):
    keys = profiles.load(repo / "profiles/qwen3.8-27b/long.env").settings
    assert keys["LLAMA_ARG_CTX_SIZE"] == "98304"
    assert not any(k.startswith("LLAMA_ARG_SPEC_") for k in keys)


REPO_PROFILES = sorted((path.parent.name, path.stem) for path in (REPO / "profiles").glob("*/*.env"))
ROLE_KEYS = {"main": "LLAMA_ARG_MODEL", "mtp": "LLAMA_ARG_SPEC_DRAFT_MODEL", "mmproj": "LLAMA_ARG_MMPROJ"}


@pytest.mark.parametrize("model_id, name", REPO_PROFILES)
def test_every_profile_loads_its_own_model_files(repo, model_id, name):
    model = pins.model(model_id)
    profile = profiles.load(repo / "profiles" / model_id / f"{name}.env")
    keys = profile.settings
    assert keys["LLAMA_ARG_ALIAS"] == model_id
    for role, key in ROLE_KEYS.items():
        if key in keys:
            assert pathlib.PurePath(keys[key]).name in {f.local_name for f in model.files if f.role == role}, key
    assert "LLAMA_ARG_MODEL" in keys
    assert (keys.get("LLAMA_ARG_SPEC_TYPE") == "draft-mtp") == ("LLAMA_ARG_SPEC_DRAFT_MODEL" in keys)
    assert profile.vram_gib <= 23.0


def test_each_model_has_its_profile_set(repo):
    catalog = profiles.load_catalog(repo / "profiles")
    assert set(catalog) == set(pins.model_ids())
    assert set(catalog["qwen3.6-35b-a3b"]) == {"agent", "long"}
    assert set(catalog["qwen3.8-27b"]) == {"agent", "long", "vision"}
    assert all("agent" in names for names in catalog.values())


def test_all_names_used_by_any_profile(repo):
    names = profiles.all_setting_names(repo / "profiles")
    assert "LLAMA_ARG_SPEC_TYPE" in names
    assert "LLAMA_ARG_MMPROJ" in names
    assert "LLAMA_ARG_MMPROJ_AUTO" in names
    assert all(n.startswith("LLAMA_ARG_") for n in names)


def test_all_names_cover_every_model_dir(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a/x.env").write_text("# qlm: vram_gib=1\nLLAMA_ARG_N_CPU_MOE=4\nLLAMA_ARG_CTX_SIZE=8\n")
    (tmp_path / "b/y.env").write_text("# qlm: vram_gib=1\nLLAMA_ARG_CTX_SIZE=8\nLLAMA_ARG_JINJA=true\n")
    assert profiles.all_setting_names(tmp_path) == ["LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_JINJA", "LLAMA_ARG_N_CPU_MOE"]


def test_all_names_are_the_union_of_every_profile_in_the_repo(repo):
    union = set()
    for path in (repo / "profiles").glob("*/*.env"):
        union.update(profiles.load(path).settings)
    assert profiles.all_setting_names(repo / "profiles") == sorted(union)


AGENT_27B_BEFORE_THE_MOVE = (
    "# qlm: vram_gib=20.3\n"
    "LLAMA_ARG_MODEL={home}/.local/share/qlm/models/Qwen3.8-27B-UD-Q4_K_M.gguf\n"
    "LLAMA_ARG_SPEC_TYPE=draft-mtp\n"
    "LLAMA_ARG_SPEC_DRAFT_MODEL={home}/.local/share/qlm/models/mtp-Qwen3.8-27B-Q4_0.gguf\n"
    "LLAMA_ARG_SPEC_DRAFT_N_MAX=3\n"
    "LLAMA_ARG_CTX_SIZE=65536\n"
    "LLAMA_ARG_N_GPU_LAYERS=all\n"
    "LLAMA_ARG_FIT=off\n"
    "LLAMA_ARG_FLASH_ATTN=on\n"
    "LLAMA_ARG_CACHE_TYPE_K=q8_0\n"
    "LLAMA_ARG_CACHE_TYPE_V=q8_0\n"
    "LLAMA_ARG_N_PARALLEL=1\n"
    "LLAMA_ARG_MMPROJ_AUTO=false\n"
    "LLAMA_ARG_CTX_CHECKPOINTS=32\n"
    "LLAMA_ARG_CACHE_RAM=16384\n"
    "LLAMA_ARG_JINJA=true\n"
    "LLAMA_ARG_THINK=auto\n"
    "LLAMA_ARG_ALIAS=qwen3.8-27b\n"
    "LLAMA_ARG_HOST=127.0.0.1\n"
    "LLAMA_ARG_PORT=9931\n"
    "LLAMA_ARG_ENDPOINT_METRICS=true\n"
    "LLAMA_ARG_ENDPOINT_SLOTS=false\n"
    "LLAMA_ARG_UI=false\n"
    "LLAMA_ARG_CORS_ORIGINS=http://qlm.invalid\n"
    "LLAMA_ARG_API_KEY_FILE={home}/.config/qlm/api-key\n"
    "LLAMA_ARG_TEMPERATURE=1.0\n"
    "LLAMA_ARG_TOP_P=0.95\n"
    "LLAMA_ARG_TOP_K=20\n"
    "LLAMA_ARG_MIN_P=0.0\n"
    "LLAMA_ARG_PRESENCE_PENALTY=0.0\n"
)


def test_the_moved_27b_agent_profile_renders_as_before(repo, tmp_path):
    home = tmp_path / "some home"
    text = profiles.load(repo / "profiles/qwen3.8-27b/agent.env").render(home)
    assert text == AGENT_27B_BEFORE_THE_MOVE.format(home=home)


@pytest.mark.parametrize(
    "text, problem",
    [
        ("LLAMA_ARG_X=1\n", "first line"),
        ("# qlm: vram_gib=20\nLLAMA_ARG_X=1\nLLAMA_ARG_X=2\n", "duplicate"),
        ("# qlm: vram_gib=20\nLLAMA_ARG_X=\n", "empty"),
        ("# qlm: vram_gib=20\nOTHER=1\n", "LLAMA_ARG_"),
    ],
)
def test_malformed_profiles_are_rejected(tmp_path, text, problem):
    p = tmp_path / "bad.env"
    p.write_text(text)
    with pytest.raises(profiles.ProfileError, match=problem):
        profiles.load(p)
