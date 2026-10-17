"""Command-line entry point. Exit codes: 0 ok, 1 a check or step failed, 2 usage error."""

import argparse
import os
import sys

from qlm import pins


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"qlm: {message}. Run qlm --help for usage.", file=sys.stderr)
        raise SystemExit(2)


def build_parser():
    parser = _Parser(prog="qlm", description="Set up and run Qwen3.8-27B with llama-server for Con.")
    sub = parser.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)
    sub.required = True
    sub.add_parser("doctor", help="check this machine before installing (read-only)")
    install = sub.add_parser("install", help="build llama.cpp and install the service, profile and API key")
    install.add_argument("--tag", default=pins.LLAMA_TAG, help=f"llama.cpp tag (only {pins.LLAMA_TAG} for now)")
    install.add_argument("--rebuild", action="store_true", help="build again even if this tag is installed")
    fetch = sub.add_parser("fetch", help="download and verify the pinned model files")
    fetch.add_argument("--vision", action="store_true", help="also fetch the vision projector")
    profile = sub.add_parser("profile", help="switch the server profile")
    profile_sub = profile.add_subparsers(dest="action", metavar="<action>", parser_class=_Parser)
    profile_sub.required = True
    use = profile_sub.add_parser("use", help="make a profile active; restarts the server if it is running")
    use.add_argument("name", help="a profile in profiles/, such as agent, long or vision")
    sub.add_parser("start", help="start the server after checking VRAM, and wait until it is ready")
    sub.add_parser("stop", help="stop the server")
    sub.add_parser("status", help="show the server's state, profile, VRAM and context use")
    sub.add_parser("smoke", help="run eight API checks against the running server")
    con = sub.add_parser("con", help="show, write or undo the Con settings for the local server")
    con_mode = con.add_mutually_exclusive_group(required=True)
    con_mode.add_argument("--print", dest="mode", action="store_const", const="print", help="show what --apply would do")
    con_mode.add_argument("--apply", dest="mode", action="store_const", const="apply", help="write the settings")
    con_mode.add_argument("--revert", dest="mode", action="store_const", const="revert", help="restore the last backup")
    suggest = sub.add_parser("suggest", help="turn Con's suggestions on (CPU-only Ollama model) or off")
    suggest_mode = suggest.add_mutually_exclusive_group(required=True)
    suggest_mode.add_argument("--enable", dest="enable", action="store_true", default=None, help="turn suggestions on")
    suggest_mode.add_argument("--disable", dest="enable", action="store_false", help="turn suggestions off")
    logs = sub.add_parser("logs", help="show the server's journal")
    logs.add_argument("-n", dest="lines", type=_positive_int, default=50, help="number of lines (default 50)")
    logs.add_argument("-f", dest="follow", action="store_true", help="keep following new lines")
    return parser


def _positive_int(text):
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return int(text)


SERVICE_COMMANDS = {"start", "stop", "status", "logs"}


def main(argv=None, ctx=None):
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    if ctx is None:
        from qlm.system import default_ctx

        ctx = default_ctx()
    try:
        return _dispatch(args, ctx)
    except BrokenPipeError:
        # Whoever read our output stopped early. Point stdout at /dev/null so the
        # interpreter's final flush doesn't raise again, and end without a message.
        if ctx.out is sys.stdout:
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            except (OSError, ValueError):
                pass
        return 1


def _dispatch(args, ctx):
    if args.command == "doctor":
        from qlm import doctor

        return doctor.run(ctx)
    if args.command == "install":
        if args.tag != pins.LLAMA_TAG:
            print(
                f"qlm: tag {args.tag} is not supported. Only {pins.LLAMA_TAG} is pinned; other tags arrive with qlm update.",
                file=ctx.err,
            )
            return 2
        from qlm import install

        return install.run(ctx, rebuild=args.rebuild)
    if args.command == "fetch":
        from qlm import fetch

        return fetch.run(ctx, vision=args.vision)
    if args.command == "profile":
        from qlm import profiles, service

        names = sorted(profiles.load_all(ctx.paths.profiles_dir))
        if args.name not in names:
            print(f"qlm: no profile named {args.name!r}. Choose one of: {', '.join(names)}.", file=ctx.err)
            return 2
        return service.run(ctx, service.profile_use, name=args.name)
    if args.command == "con":
        from qlm import con, service

        command = {"print": con.print_plan, "apply": con.apply, "revert": con.revert}[args.mode]
        return service.run(ctx, command)
    if args.command == "suggest":
        from qlm import con, service

        return service.run(ctx, con.suggest, enable=args.enable)
    if args.command == "smoke":
        from qlm import smoke

        return smoke.run(ctx)
    if args.command in SERVICE_COMMANDS:
        from qlm import service

        if args.command == "logs":
            return service.logs(ctx, lines=args.lines, follow=args.follow)
        return service.run(ctx, getattr(service, args.command))
    return 2
