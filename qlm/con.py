"""`qlm con` and `qlm suggest`: point Con at the local server, and opt in to suggestions from Ollama.

Con reads ~/.config/con/con.conf only at startup, and an unknown `con.*` key stops it from
loading the file at all (ADR 0003). So these commands refuse while Con runs, write only the
keys below, back the file up first, write it atomically, and check the result.
"""

import os
import re
import tempfile
from pathlib import Path

from qlm import files, service

CON_HEADER = "# Con configuration\ncon.version = 1\n"
DASHED_PREFIX = "con.agent.providers.openai-compatible."
TEMPERATURE = "con.agent.temperature"
ENABLED = "con.agent.suggestion_model.enabled"
SUGGEST_PROVIDER = "con.agent.suggestion_model.provider"
SUGGEST_MODEL = "con.agent.suggestion_model.model"
API_KEY = "con.agent.providers.openaicompatible.api_key"
LEGACY_FILES = ("config.ghostty", "config.toml")
BACKUP = re.compile(r"^con\.conf\.(\d{8}T\d{6}Z)(?:-(\d+))?\.(bak|absent)$")
SUGGEST_BASE = "qwen2.5-coder:1.5b"
SUGGEST_NAME = "con-suggest"
CPU_ONLY = "PARAMETER num_gpu 0"
TIMEOUT_S = 30


def _server_keys(api_key):
    return [
        ("con.agent.provider", '"openaicompatible"'),
        ("con.agent.provider_is_explicit", "true"),
        ("con.agent.providers.openaicompatible.base_url", '"http://127.0.0.1:9931/v1"'),
        ("con.agent.providers.openaicompatible.model", '"qwen3.8-27b"'),
        (API_KEY, f'"{api_key}"'),
        (ENABLED, "false"),
    ]


SUGGEST_ON = [(ENABLED, "true"), (SUGGEST_PROVIDER, '"ollama"'), (SUGGEST_MODEL, f'"{SUGGEST_NAME}"')]
SUGGEST_OFF = [(ENABLED, "false")]


# The text editor. Lines are split on "\n" only, so a "\r" stays on lines that aren't touched.


def holds(line, key):
    """Whether this line sets `key`, as Con reads it: trimmed, not a comment, key before the first '='."""
    stripped = line.strip()
    if stripped.startswith("#") or "=" not in stripped:
        return False
    return stripped.split("=", 1)[0].strip() == key


def decode(value_text):
    value = value_text.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def current_value(lines, key):
    """The value Con uses for `key`: the last line holding it, trimmed, outer quotes removed."""
    for line in reversed(lines):
        if holds(line, key):
            return decode(line.split("=", 1)[1])
    return None


def edit(text, settings, remove=()):
    """Set each (key, value_text) in place or at the end, and drop every line holding a key in `remove`."""
    lines = text.split("\n")
    trailing = lines[-1] == ""
    if trailing:
        lines.pop()
    done = set()
    out = []
    for line in lines:
        key = next((k for k, _ in settings if holds(line, k)), None)
        if key is not None:
            if key in done:
                continue
            done.add(key)
            value = dict(settings)[key]
            out.append(f"{key} = {value}" + ("\r" if line.endswith("\r") else ""))
        elif any(holds(line, k) for k in remove):
            continue
        else:
            out.append(line)
    new = "\n".join(out) + ("\n" if out and trailing else "")
    appended = [f"{k} = {v}\n" for k, v in settings if k not in done]
    if appended and new and not new.endswith("\n"):
        new += "\n"
    return new + "".join(appended)


# Checks shared by every command that writes


def _check_xdg(ctx):
    xdg = ctx.env.xdg_config_home()
    if xdg is not None and os.path.normpath(xdg) != os.path.normpath(ctx.paths.home / ".config"):
        raise service.Stop(
            f"XDG_CONFIG_HOME is set to {xdg}, so Con reads its config from there, not {ctx.paths.con_conf}",
            "Unset XDG_CONFIG_HOME or edit Con's config by hand",
        )


def _check_con_closed(ctx):
    r = ctx.runner.run(["pgrep", "-x", "con"], timeout=TIMEOUT_S)
    if r.code == 1:
        return
    if r.code == 0:
        pids = ", ".join(r.stdout.split())
        raise service.Stop(f"Con is running (PID {pids})", "quit Con first, then run this again")
    raise service.Stop(f"couldn't tell whether Con is running (pgrep exit {r.code})", "Check that pgrep works")


def _target(ctx):
    """The file to edit: con.conf, or its symlink's target, which must be inside the home directory."""
    conf = ctx.paths.con_conf
    if not conf.is_symlink():
        return conf
    target = conf.resolve()
    if not target.is_relative_to(ctx.paths.home.resolve()):
        raise service.Stop(f"{conf} links to {target}, outside your home directory", "Edit that file by hand")
    return target


def _read(ctx, target):
    """(text or None, original bytes or None). Refuses when Con would still migrate an older file."""
    if target.exists():
        data = target.read_bytes()
        try:
            return data.decode("utf-8"), data
        except UnicodeDecodeError:
            raise service.Stop(f"{target} is not valid UTF-8", "Fix the file by hand") from None
    legacy = [name for name in LEGACY_FILES if (ctx.paths.con_conf.parent / name).exists()]
    if legacy:
        raise service.Stop(
            f"{ctx.paths.con_conf} doesn't exist yet, but {legacy[0]} does, and Con migrates it only while con.conf is absent",
            "Start Con once and quit it, so it creates con.conf, then run this again",
        )
    return None, None


def _check_dashed(text, target):
    for n, line in enumerate((text or "").split("\n"), start=1):
        if line.lstrip().startswith(DASHED_PREFIX):
            raise service.Stop(
                f"line {n} of {target} uses {DASHED_PREFIX}*, which stops Con loading its config",
                "Rename those keys to con.agent.providers.openaicompatible.* and run this again",
            )


def _api_key(ctx):
    p = ctx.paths.api_key
    key = p.read_text().strip() if p.exists() else ""
    if not key:
        raise service.Stop(f"the API key file {p} is missing or empty", "Run qlm install")
    return key


def _stamp(ctx):
    return ctx.clock.now().strftime("%Y%m%dT%H%M%SZ")


def _backups_dir(ctx):
    d = ctx.paths.backups
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def _write(path, data, mode):
    files.replace_atomically(path, data, mode)


def _verify(target, settings, expected_values):
    lines = target.read_text().split("\n")
    for key, _ in settings:
        if sum(holds(l, key) for l in lines) != 1 or current_value(lines, key) != expected_values[key]:
            return f"{key} is not set as expected"
    if any(holds(l, TEMPERATURE) for l in lines):
        return f"{TEMPERATURE} is still set"
    return None


def _apply(ctx, settings):
    """Edit con.conf: back up, write atomically, verify, and roll back on a bad result. Returns True if it changed."""
    target = _target(ctx)
    text, original = _read(ctx, target)
    _check_dashed(text, target)
    new = edit(text if text is not None else CON_HEADER, settings, remove=(TEMPERATURE,))
    if text is not None and new == text:
        return False
    stamp = _stamp(ctx)
    d = _backups_dir(ctx)
    if original is not None:
        files.create_new(d, lambda n: f"con.conf.{files.stamped(stamp, n)}.bak", original)
        mode = os.stat(target).st_mode & 0o7777
    else:
        files.create_new(d, lambda n: f"con.conf.{files.stamped(stamp, n)}.absent")
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = 0o600
    _write(target, new.encode(), mode)
    problem = _verify(target, settings, {k: decode(v) for k, v in settings})
    if problem:
        if original is not None:
            files.replace_atomically(target, original, mode)
        else:
            target.unlink()
        raise service.Stop(f"the written file didn't check out ({problem}), so it was put back", "Report this as a bug")
    return True


# qlm con


def print_plan(ctx):
    target = ctx.paths.con_conf
    lines = target.read_text(errors="replace").split("\n") if target.exists() else []
    key = ctx.paths.api_key.read_text().strip() if ctx.paths.api_key.exists() else None
    for k, v in _plan_keys(lines, key or ""):
        have = current_value(lines, k)
        if k == API_KEY and key is None:
            state = "unknown: no API key file"
        else:
            state = "missing" if have is None else "same" if have == decode(v) else "different"
        shown = "<hidden>" if k == API_KEY else v
        ctx.say(f"{k} = {shown}  ({state})")
    if any(holds(l, TEMPERATURE) for l in lines):
        ctx.say(f"{TEMPERATURE}  (will be removed)")
    return 0


def _suggest_uses_ollama(lines):
    return current_value(lines, SUGGEST_PROVIDER) == "ollama" and current_value(lines, SUGGEST_MODEL) == SUGGEST_NAME


def _plan_keys(lines, api_key):
    keys = _server_keys(api_key)
    if _suggest_uses_ollama(lines) and current_value(lines, ENABLED) == "true":
        keys = [kv for kv in keys if kv[0] != ENABLED]
    return keys


def apply(ctx):
    _check_xdg(ctx)
    _check_con_closed(ctx)
    key = _api_key(ctx)
    target = _target(ctx)
    lines = target.read_text(errors="replace").split("\n") if target.exists() else []
    if not _apply(ctx, _plan_keys(lines, key)):
        ctx.say("Con is already configured for the local server.")
        return 0
    ctx.say(f"Con now points at qwen3.8-27b on 127.0.0.1:9931 (backup in {ctx.paths.backups}).")
    ctx.say("Con picks up the change the next time it starts. Checking that its agent answers is a manual step.")
    if not service.unit_state(ctx).running:
        ctx.say("The server isn't running; start it with qlm start.")
    return 0


def _newest_backup(ctx):
    d = ctx.paths.backups
    found = []
    for p in d.iterdir() if d.exists() else []:
        m = BACKUP.match(p.name)
        if m:
            found.append(((m.group(1), int(m.group(2) or 0)), p))
    return max(found)[1] if found else None


def revert(ctx):
    _check_xdg(ctx)
    _check_con_closed(ctx)
    backup = _newest_backup(ctx)
    if backup is None:
        raise service.Stop(f"no con.conf backup in {ctx.paths.backups}", "Nothing to revert")
    target = _target(ctx)
    stamp = _stamp(ctx)
    d = _backups_dir(ctx)
    if target.exists():
        files.create_new(d, lambda n: f"con.conf.{files.stamped(stamp, n)}.pre-revert", target.read_bytes())
    if backup.name.endswith(".absent"):
        if target.exists():
            target.unlink()
        ctx.say(f"Removed {ctx.paths.con_conf}; there was none before qlm changed it.")
    else:
        mode = os.stat(target).st_mode & 0o7777 if target.exists() else 0o600
        target.parent.mkdir(parents=True, exist_ok=True)
        _write(target, backup.read_bytes(), mode)
        ctx.say(f"Restored {ctx.paths.con_conf} from {backup.name}.")
    ctx.say("Con picks up the change the next time it starts.")
    return 0


# qlm suggest


def _ollama_names(stdout):
    names = set()
    for line in stdout.splitlines()[1:]:
        if line.strip():
            name = line.split()[0]
            names.add(name if ":" in name else f"{name}:latest")
    return names


def _prepare_ollama(ctx):
    if not ctx.runner.which("ollama"):
        raise service.Stop("Ollama is not installed", "Install Ollama, or leave suggestions off")
    r = ctx.runner.run(["ollama", "list"], timeout=TIMEOUT_S)
    if r.code != 0:
        raise service.Stop(f"ollama list failed: {r.output.strip()}", "Start the Ollama service (systemctl start ollama)")
    names = _ollama_names(r.stdout)
    if f"{SUGGEST_NAME}:latest" in names:
        shown = ctx.runner.run(["ollama", "show", SUGGEST_NAME, "--modelfile"], timeout=TIMEOUT_S)
        if CPU_ONLY not in [l.strip() for l in shown.stdout.splitlines()]:
            raise service.Stop(
                f"the existing Ollama model {SUGGEST_NAME} isn't limited to the CPU, so it could take VRAM",
                f"Remove it with: ollama rm {SUGGEST_NAME}, then run this again",
            )
        return
    if SUGGEST_BASE not in names:
        ctx.say(f"Downloading {SUGGEST_BASE} (about 1 GB)...")
        code = ctx.runner.passthrough(["ollama", "pull", SUGGEST_BASE])
        if code != 0:
            raise service.Stop(f"ollama pull {SUGGEST_BASE} failed (exit {code})", "Check Ollama and try again")
    with tempfile.TemporaryDirectory(prefix="qlm-suggest-") as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text(f"FROM {SUGGEST_BASE}\n{CPU_ONLY}\n")
        r = ctx.runner.run(["ollama", "create", SUGGEST_NAME, "-f", modelfile], timeout=300)
    if r.code != 0:
        raise service.Stop(f"ollama create {SUGGEST_NAME} failed: {r.output.strip()}", "Check Ollama and try again")


def suggest(ctx, enable):
    _check_xdg(ctx)
    _check_con_closed(ctx)
    if enable:
        target = _target(ctx)
        text, _ = _read(ctx, target)
        _check_dashed(text, target)
        _prepare_ollama(ctx)
    changed = _apply(ctx, SUGGEST_ON if enable else SUGGEST_OFF)
    state = "on, from Ollama's CPU-only con-suggest" if enable else "off"
    if not changed:
        ctx.say(f"Con's suggestions are already {state}.")
        return 0
    ctx.say(f"Con's suggestions are now {state}. Con picks up the change the next time it starts.")
    return 0
