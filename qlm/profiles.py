"""Server profiles: LLAMA_ARG_* env files with a `# qlm: vram_gib=<n>` first line."""

import re
from dataclasses import dataclass
from pathlib import Path

HEADER = re.compile(r"^# qlm: vram_gib=(\d+(?:\.\d+)?)$")
HOME_MARKER = "%h"


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    vram_gib: float
    lines: tuple
    settings: dict

    def render(self, home):
        """The env file with the home marker expanded, as systemd will read it."""
        return "\n".join(line.replace(HOME_MARKER, str(home)) for line in self.lines) + "\n"


def parse(name, text):
    lines = text.splitlines()
    m = HEADER.match(lines[0]) if lines else None
    if not m:
        raise ProfileError(f"{name}: first line must be '# qlm: vram_gib=<number>'")
    settings = {}
    for n, line in enumerate(lines[1:], start=2):
        key, sep, value = line.partition("=")
        if not sep or not key.startswith("LLAMA_ARG_"):
            raise ProfileError(f"{name} line {n}: expected LLAMA_ARG_<NAME>=<value>")
        if not value:
            raise ProfileError(f"{name} line {n}: empty value for {key}")
        if key in settings:
            raise ProfileError(f"{name} line {n}: duplicate {key}")
        settings[key] = value
    return Profile(name=name, vram_gib=float(m.group(1)), lines=tuple(lines), settings=settings)


def load(path):
    path = Path(path)
    return parse(path.stem, path.read_text())


def load_all(profiles_dir):
    return {p.stem: load(p) for p in sorted(Path(profiles_dir).glob("*.env"))}


def all_setting_names(profiles_dir):
    names = set()
    for profile in load_all(profiles_dir).values():
        names.update(profile.settings)
    return sorted(names)
