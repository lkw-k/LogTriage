"""configs/base.yaml 로더. 설정의 정본은 YAML 한 곳이다."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/base.yaml"


def load(path=DEFAULT_CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
