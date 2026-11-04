import ast
import re

from tests.conftest import REPO

FORBIDDEN = "/" + "home/"
W = "c" "on"
PATTERN = re.compile(rf"\b{W}\b|{W}-terminal|{W}\.conf|{W}\.agent|" + "now" "ledge", re.I)
TRACKED_FILES = ("README.md", "pyproject.toml", ".gitignore")
TRACKED_DIRS = ("bin", "docs", "profiles", "qlm", "systemd", "tests")


def test_no_test_file_or_fixture_contains_a_home_path():
    offenders = [
        str(p.relative_to(REPO))
        for p in (REPO / "tests").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and FORBIDDEN.encode() in p.read_bytes()
    ]
    assert offenders == []


def test_no_fixture_holds_an_api_key_shaped_token():
    token = re.compile(rb"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")
    fake = b"k" * 43
    found = [
        f"{p.relative_to(REPO)}: {m.decode()}"
        for p in (REPO / "tests/fixtures").rglob("*")
        if p.is_file()
        for m in token.findall(p.read_bytes())
        if m != fake
    ]
    assert found == []


def test_code_imports_only_the_standard_library():
    import sys

    allowed = set(sys.stdlib_module_names) | {"qlm"}
    third_party = []
    for path in (REPO / "qlm").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            third_party += [f"{path.name}: {n}" for n in names if n.split(".")[0] not in allowed]
    assert third_party == []


def test_no_sudo_anywhere_in_the_code():
    hits = [p.name for p in (REPO / "qlm").glob("*.py") if "sudo" in p.read_text()]
    hits += ["bin/qlm"] if "sudo" in (REPO / "bin/qlm").read_text() else []
    assert hits == []


def _tracked_text_files():
    paths = [REPO / name for name in TRACKED_FILES]
    paths += [p for d in TRACKED_DIRS for p in sorted((REPO / d).rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    for path in paths:
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def test_no_con_mentions():
    hits = [
        f"{path.relative_to(REPO)}:{n}"
        for path, text in _tracked_text_files()
        for n, line in enumerate(text.splitlines(), start=1)
        if PATTERN.search(line)
    ]
    assert hits == []


def test_con_pattern_self_check():
    for text in ("C" "on's agent", W + ".conf", "now" "ledge"):
        assert PATTERN.search(text), text
    for text in ("config", "context", "second", "icon"):
        assert not PATTERN.search(text), text


def test_descriptions_name_no_model():
    from qlm import cli, pins

    unit = (REPO / "systemd/qlm-server.service").read_text()
    descriptions = [line for line in unit.splitlines() if line.startswith("Description=")]
    descriptions.append(cli.build_parser().description)
    for text in descriptions:
        assert "qwen" not in text.lower(), text
        assert not any(model_id in text for model_id in pins.model_ids()), text


def test_every_adr_link_resolves():
    link = re.compile(r"\]\(([^)#\s]+\.md)\)")
    missing = []
    for doc in (REPO / "README.md", REPO / "docs/architecture.md"):
        for target in link.findall(doc.read_text()):
            path = (doc.parent / target).resolve()
            if path.parent == REPO / "docs/adr" and not path.is_file():
                missing.append(f"{doc.relative_to(REPO)}: {target}")
    assert missing == []
