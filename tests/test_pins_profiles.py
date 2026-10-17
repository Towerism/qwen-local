import re

import pytest

from qlm import pins, profiles


def test_pins_match_the_values_written_in_the_adrs(repo):
    adr1 = (repo / "docs/adr/0001-runtime-llama-cpp-server.md").read_text()
    adr2 = (repo / "docs/adr/0002-model-file-and-vram-budget.md").read_text()

    assert f"tag `{pins.LLAMA_TAG}`, commit `{pins.LLAMA_COMMIT}`" in adr1
    assert f"`{pins.HF_REPO}` at revision `{pins.HF_REVISION}`" in adr2
    rows = {
        m.group(1): (int(m.group(2).replace(",", "")), m.group(3))
        for m in re.finditer(r"^\| `([^`]+)`(?: \([^)]*\))? \| ([\d,]+) \| `([0-9a-f]{64})` \|$", adr2, re.M)
    }
    assert {f.repo_path: (f.size, f.sha256) for f in pins.FILES} == rows


def test_only_the_projector_is_vision_only():
    assert [f.local_name for f in pins.files()] == ["Qwen3.8-27B-UD-Q4_K_M.gguf", "mtp-Qwen3.8-27B-Q4_0.gguf"]
    assert len(pins.files(vision=True)) == 3


@pytest.mark.parametrize("name, vram", [("agent", 20.3), ("long", 20.0), ("vision", 18.7)])
def test_profiles_carry_their_vram_need(repo, name, vram):
    profile = profiles.load(repo / "profiles" / f"{name}.env")
    assert profile.vram_gib == vram


def test_rendering_replaces_home_marker(repo, tmp_path):
    profile = profiles.load(repo / "profiles/agent.env")
    text = profile.render(tmp_path / "some home")
    assert "%h" not in text
    assert f"LLAMA_ARG_MODEL={tmp_path}/some home/.local/share/qlm/models/Qwen3.8-27B-UD-Q4_K_M.gguf" in text
    assert text.splitlines()[0] == "# qlm: vram_gib=20.3"


def test_vision_profile_loads_the_projector_and_drops_the_no_projector_line(repo):
    keys = profiles.load(repo / "profiles/vision.env").settings
    assert "LLAMA_ARG_MMPROJ" in keys
    assert "LLAMA_ARG_MMPROJ_AUTO" not in keys
    assert not any(k.startswith("LLAMA_ARG_SPEC_") for k in keys)


def test_long_profile_has_no_mtp_and_96k_context(repo):
    keys = profiles.load(repo / "profiles/long.env").settings
    assert keys["LLAMA_ARG_CTX_SIZE"] == "98304"
    assert not any(k.startswith("LLAMA_ARG_SPEC_") for k in keys)


def test_all_names_used_by_any_profile(repo):
    names = profiles.all_setting_names(repo / "profiles")
    assert "LLAMA_ARG_SPEC_TYPE" in names
    assert "LLAMA_ARG_MMPROJ" in names
    assert "LLAMA_ARG_MMPROJ_AUTO" in names
    assert all(n.startswith("LLAMA_ARG_") for n in names)


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
