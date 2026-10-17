import ast

from tests.conftest import REPO

FORBIDDEN = "/" + "home/"


def test_no_test_file_or_fixture_contains_a_home_path():
    offenders = [
        str(p.relative_to(REPO))
        for p in (REPO / "tests").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and FORBIDDEN.encode() in p.read_bytes()
    ]
    assert offenders == []


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
