"""`qlm update --tag <bN> --commit <sha>`: move to a newer llama.cpp tag, and back if it fails smoke.

The new tag is built beside the old one, and `current` moves only after the build, the --help
check and qlm start's checks pass (ADR 0001: a bad build never replaces a working one).
"""

from qlm import install, interrupts, pins, service, smoke

WARNING = "Note: qlm update restarts the server, which interrupts a Con turn in progress."
INTERRUPTED = "interrupted; nothing was changed"


def run(ctx, tag, commit):
    try:
        return _run(ctx, tag, commit)
    except service.Stop as stop:
        return ctx.fail(stop.what, stop.todo)
    except KeyboardInterrupt:
        return ctx.fail(INTERRUPTED, "Run qlm update again when you're ready")
    except BrokenPipeError:
        raise
    except OSError as e:
        return ctx.fail(f"file error: {e}", "Fix the cause and run qlm update again")


def old_build(ctx):
    """The directory `current` points at, if it is a build directly under ~/.local/opt/llama.cpp."""
    current = ctx.paths.llama_current
    if not current.is_symlink():
        return None
    target = current.resolve()
    if target.parent != ctx.paths.llama_opt.resolve() or not (target / "bin/llama-server").is_file():
        return None
    return ctx.paths.llama_opt / target.name


def leftover_bench_backup(ctx):
    p = ctx.paths
    if p.bench_backup.exists():
        raise service.Stop(
            f"{p.bench_backup} is left from a qlm bench that was killed before it restored the profile",
            f"Restore it with: mv {p.bench_backup} {p.active_env}, then run qlm stop and qlm start",
        )


def _run(ctx, tag, commit):
    p = ctx.paths
    old = old_build(ctx)
    if old is None:
        raise service.Stop(f"{p.llama_current} doesn't lead to an installed llama-server build", "Run qlm install")
    if not p.unit_file.exists():
        raise service.Stop("the qlm-server unit is not installed", "Run qlm install")
    leftover_bench_backup(ctx)
    if tag == old.name:
        ctx.say(f"llama.cpp is already on {tag}.")
        return 0
    ctx.say(WARNING)

    with interrupts.sigterm_as_interrupt():
        install.build_tag(ctx, tag, commit, "update", "--commit is")
        state = service.unit_state(ctx)
        was_running = state.running
        profile = service.preflight(ctx, check_port=not was_running)
        service.check_ollama(ctx)
        credit = (service.server_vram_mib(ctx, state.pid) or 0) if was_running else 0
        service.check_vram(ctx, profile, credit_mib=credit)

        install.point_current(ctx, p.tag_dir(tag))
        try:
            failure = _start_and_smoke(ctx, tag, was_running)
        except KeyboardInterrupt:
            failure = f"interrupted while testing {tag}"
    if failure:
        return _roll_back(ctx, old, tag, was_running, failure)

    if not was_running:
        service.stop(ctx)
    ctx.say(f"llama.cpp moved from {old.name} to {tag}, and smoke passed.")
    ctx.say(f"ADR 0001 and qlm/pins.py still pin {pins.LLAMA_TAG}; update both by hand to make {tag} the pinned tag.")
    return 0


def _start_and_smoke(ctx, tag, was_running):
    """Bring the server up on the new build and run smoke. Returns what failed, or None."""
    if not service.bring_up(ctx, restart=was_running):
        return f"the server didn't become ready on {tag}"
    failed = []
    if smoke.run(ctx, failed) != 0:
        return f"smoke failed on {tag} ({failed[0] if failed else 'see the smoke output above'})"
    return None


def _roll_back(ctx, old, tag, was_running, failure):
    with interrupts.deferred():
        install.point_current(ctx, old)
        if was_running:
            if not service.bring_up(ctx, restart=True):
                return ctx.fail(
                    f"{failure}, and {old.name} didn't become ready either, so the server is stopped",
                    "Run qlm start, and qlm logs to see why",
                )
        else:
            service.stop(ctx)
    return ctx.fail(
        f"{failure}; current is back on {old.name}",
        f"The {tag} build is kept in {ctx.paths.tag_dir(tag)} for inspection",
    )
