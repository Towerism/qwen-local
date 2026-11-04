"""Which model is selected: the selection file, else active.env's alias, else the default."""

import pytest

from qlm import profiles, selection, service
from tests.fakes import make_ctx, select_model


def write_active_env(ctx, repo, model_id):
    ctx.paths.config_dir.mkdir(parents=True, exist_ok=True)
    text = profiles.load(repo / "profiles" / model_id / "agent.env").render(ctx.paths.home)
    ctx.paths.active_env.write_text(text)


def test_a_fresh_home_selects_the_35b_model(home, repo):
    assert selection.current(make_ctx(home, repo)) == "qwen3.6-35b-a3b"


def test_the_selection_file_wins_over_the_active_env_alias(home, repo):
    ctx = make_ctx(home, repo)
    write_active_env(ctx, repo, "qwen3.8-27b")
    select_model(home, "qwen3.6-35b-a3b")
    assert selection.current(ctx) == "qwen3.6-35b-a3b"


def test_an_install_from_before_selection_keeps_its_model_without_writing_the_file(home, repo):
    ctx = make_ctx(home, repo)
    write_active_env(ctx, repo, "qwen3.8-27b")
    assert selection.current(ctx) == "qwen3.8-27b"
    assert not ctx.paths.selection.exists()


def test_an_alias_outside_the_catalog_falls_back_to_the_default(home, repo):
    ctx = make_ctx(home, repo)
    ctx.paths.config_dir.mkdir(parents=True)
    ctx.paths.active_env.write_text("# qlm: vram_gib=1\nLLAMA_ARG_ALIAS=my-model\n")
    assert selection.current(ctx) == "qwen3.6-35b-a3b"


def test_an_unknown_id_in_the_selection_file_names_the_file_and_the_valid_ids(home, repo):
    ctx = make_ctx(home, repo)
    select_model(home, "qwen9-1t")
    with pytest.raises(service.Stop) as e:
        selection.current(ctx)
    assert str(ctx.paths.selection) in e.value.what
    assert "qwen3.6-35b-a3b, qwen3.8-27b" in e.value.what
