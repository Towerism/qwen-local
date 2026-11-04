"""The selected catalog model: `~/.config/qlm/model`, else the model active.env runs, else the default.

The unit only reads active.env. The selection decides which model's profiles qlm renders into it,
so a model can be selected, and its files fetched, while the server keeps running the previous one.
"""

from qlm import files, pins, service


def active_alias(ctx):
    """The LLAMA_ARG_ALIAS of active.env, or None when the file or the line is missing."""
    p = ctx.paths
    if not p.active_env.exists():
        return None
    for line in p.active_env.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key == "LLAMA_ARG_ALIAS":
            return value
    return None


def current(ctx):
    """The selected model id. Raises Stop when the selection file names a model not in the catalog."""
    p = ctx.paths
    if p.selection.exists():
        model_id = p.selection.read_text().strip()
        if model_id not in pins.model_ids():
            ids = ", ".join(pins.model_ids())
            raise service.Stop(f"{p.selection} names {model_id!r}, which is not one of: {ids}", "Run qlm use <model>")
        return model_id
    alias = active_alias(ctx)  # an install from before selection existed keeps the model it runs
    return alias if alias in pins.model_ids() else pins.DEFAULT_MODEL


def save(ctx, model_id):
    p = ctx.paths
    p.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    files.replace_atomically(p.selection, f"{model_id}\n".encode(), 0o600)
