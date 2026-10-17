"""Every file location qlm uses, derived from one home directory (docs/architecture.md)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    home: Path
    repo: Path

    def __post_init__(self):
        object.__setattr__(self, "home", Path(self.home))
        object.__setattr__(self, "repo", Path(self.repo))

    @property
    def llama_src(self):
        return self.home / ".cache/qlm/llama.cpp"

    def build_dir(self, tag):
        return self.home / ".cache/qlm/build" / tag

    @property
    def llama_opt(self):
        return self.home / ".local/opt/llama.cpp"

    def tag_dir(self, tag):
        return self.llama_opt / tag

    @property
    def llama_current(self):
        return self.llama_opt / "current"

    @property
    def models(self):
        return self.home / ".local/share/qlm/models"

    @property
    def config_dir(self):
        return self.home / ".config/qlm"

    @property
    def active_env(self):
        return self.config_dir / "active.env"

    @property
    def api_key(self):
        return self.config_dir / "api-key"

    @property
    def unit_file(self):
        return self.home / ".config/systemd/user/qlm-server.service"

    @property
    def state_dir(self):
        return self.home / ".local/state/qlm"

    @property
    def logs(self):
        return self.state_dir / "logs"

    @property
    def backups(self):
        return self.state_dir / "backups"

    @property
    def bench(self):
        return self.state_dir / "bench"

    @property
    def local_bin(self):
        return self.home / ".local/bin"

    @property
    def qlm_link(self):
        return self.local_bin / "qlm"

    @property
    def con_conf(self):
        return self.home / ".config/con/con.conf"

    @property
    def profiles_dir(self):
        return self.repo / "profiles"

    @property
    def unit_template(self):
        return self.repo / "systemd/qlm-server.service"

    @property
    def entry_point(self):
        return self.repo / "bin/qlm"
