"""Command-line entry point. Exit codes: 0 ok, 1 a check or step failed, 2 usage error."""

import argparse
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
    return parser


def main(argv=None, ctx=None):
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    if ctx is None:
        from qlm.system import default_ctx

        ctx = default_ctx()

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
    return 2
