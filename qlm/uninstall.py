"""`qlm uninstall [--models] [--yes]`: remove what qlm install put in place, after listing it and asking.

It keeps ~/.local/state/qlm/ (bench results, logs and the con.conf backups qlm con --revert needs),
and the model files unless --models is given. It never edits con.conf.
"""

import os
import re
import shutil

from qlm import con, service

BUILD_NAME = re.compile(r"b[0-9]+")
CON_BASE_URL = "http://127.0.0.1:9931/v1"
PROMPT = "Type yes to remove these:"


def run(ctx, models=False, yes=False):
    try:
        return _run(ctx, models, yes)
    except service.Stop as stop:
        return ctx.fail(stop.what, stop.todo)


def _builds(p):
    if not p.llama_opt.is_dir() or p.llama_opt.is_symlink():
        return []
    return sorted(e for e in p.llama_opt.iterdir() if BUILD_NAME.fullmatch(e.name) and (e.is_dir() or e.is_symlink()))


def _our_link(p):
    link = p.qlm_link
    return link.is_symlink() and os.path.realpath(link) == os.path.realpath(p.entry_point)


def plan(ctx, models):
    """The steps in removal order, as (label, [paths]), with only the paths that exist; and what is kept."""
    p = ctx.paths
    exists = os.path.lexists
    steps = [
        ("unit", [p.unit_file] if exists(p.unit_file) else []),
        ("builds", ([p.llama_current] if exists(p.llama_current) else []) + _builds(p)),
        ("cache", [p.home / ".cache/qlm"] if exists(p.home / ".cache/qlm") else []),
        ("link", [p.qlm_link] if _our_link(p) else []),
        ("config", [p.config_dir] if exists(p.config_dir) else []),
        ("models", [p.models] if models and exists(p.models) else []),
    ]
    keep = [
        (p.state_dir, "bench results, logs, and the con.conf backups that qlm con --revert restores"),
    ]
    if not models:
        keep.append((p.models, "the model files (add --models to remove them)"))
    keep += [
        (p.repo, "this repo"),
        (p.con_conf, "Con's settings, which still hold the API key; harmless once the server is gone"),
        ("con-suggest", "Ollama's model, if qlm suggest --enable created it"),
    ]
    if exists(p.qlm_link) and not _our_link(p):
        keep.append((p.qlm_link, "it isn't a link to this repo's bin/qlm"))
    return steps, keep


def _con_points_here(ctx):
    try:
        lines = ctx.paths.con_conf.read_text().split("\n")
    except OSError:
        return False
    return (con.current_value(lines, "con.agent.provider") == "openaicompatible"
            and con.current_value(lines, "con.agent.providers.openaicompatible.base_url") == CON_BASE_URL)


def _run(ctx, models, yes):
    state = service.unit_state(ctx)
    steps, keep = plan(ctx, models)
    paths = [path for _, group in steps for path in group]
    if not paths and state.active == "inactive":
        ctx.say("Nothing to remove.")
        return 0

    ctx.say("Will remove:")
    if state.running:
        ctx.say("  the running qlm-server (stopped first)")
    for path in paths:
        ctx.say(f"  {path}")
    ctx.say("Will keep:")
    for what, why in keep:
        ctx.say(f"  {what}: {why}")
    if _con_points_here(ctx):
        ctx.say(f"Con is still set to use the local server. Restore your earlier Con settings with: "
                f"{ctx.paths.entry_point} con --revert")

    if not yes:
        if not ctx.env.stdin_is_tty():
            ctx.warn("qlm: uninstall needs confirmation. Run it in a terminal, or pass --yes")
            return 2
        try:
            answer = ctx.env.ask(PROMPT)
        except (KeyboardInterrupt, EOFError):
            answer = ""
        if answer.strip() != "yes":
            return ctx.fail("uninstall cancelled; nothing was removed", "Run qlm uninstall again to remove them")

    if state.running:
        r = service.systemctl(ctx, "stop", service.UNIT)
        if r.code != 0:
            raise service.Stop(f"systemctl --user stop failed: {r.output.strip()}", "Nothing was removed; stop it and try again")
    service.systemctl(ctx, "reset-failed", service.UNIT)

    removed = []
    for label, group in steps:
        try:
            for path in group:
                _remove(path)
                removed.append(path)
                if label == "unit":
                    _reload(ctx)
            if label == "builds":
                _rmdir_if_empty(ctx.paths.llama_opt)
            if label == "models":
                _rmdir_if_empty(ctx.paths.models.parent)
        except OSError as e:
            return _partial(ctx, removed, [x for x in paths if x not in removed], e)
    ctx.say("Removed. The repo, the state directory and anything listed above as kept are still there.")
    return 0


def _remove(path):
    """Remove a file, a link (never what it points at), or a directory tree without following links in it."""
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _reload(ctx):
    r = service.systemctl(ctx, "daemon-reload")
    if r.code != 0:
        ctx.warn(f"warning: systemctl --user daemon-reload failed: {r.output.strip()}")


def _rmdir_if_empty(path):
    try:
        path.rmdir()
    except OSError:
        pass  # missing, or holds something qlm didn't make


def _partial(ctx, removed, remaining, error):
    ctx.warn("Removed:")
    for path in removed:
        ctx.warn(f"  {path}")
    ctx.warn("Still there:")
    for path in remaining:
        ctx.warn(f"  {path}")
    return ctx.fail(f"uninstall stopped: {error}", "Fix the cause and run qlm uninstall again to remove the rest")
