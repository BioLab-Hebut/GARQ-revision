"""Portable locations only; importing this helper does not run an analysis."""
from pathlib import Path
import json, os, re
from settings_io import read_settings

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def env_path(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Set {name} to the required external input path; see README.md")
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$\{[^}]+\}", value):
        raise RuntimeError(f"Unresolved environment variable in {name}: {value}")
    return Path(value).resolve()

def output_root():
    p = env_path("GARQ_REVISION_OUTPUT", PACKAGE_ROOT / "results")
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_config(name):
    folder = env_path("GARQ_V2_CONFIG_DIR", PACKAGE_ROOT / "configs")
    path=folder/name
    if not path.exists() and path.suffix==".py" and path.with_suffix(".json").exists():
        path=path.with_suffix(".json")
    value=read_settings(path)
    def expand(item):
        if isinstance(item, dict): return {k: expand(v) for k, v in item.items()}
        if isinstance(item, list): return [expand(v) for v in item]
        if isinstance(item, str):
            result = os.path.expandvars(os.path.expanduser(item))
            if re.search(r"\$\{[^}]+\}", result):
                raise RuntimeError(f"Unresolved input path in {name}: {result}")
            return result
        return item
    return expand(value)
