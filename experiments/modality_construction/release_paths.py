"""Load input configurations and locate the shared GARQ implementation."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re

PACKAGE_ROOT = Path(__file__).resolve().parent
BACKEND_HASHES = {'GARQ.py': '49c97de6440cd8ed76eca677a9225214752856461cc27930a0d0a465ae3179ab',
 'data_utils.py': 'f86c775960003b070d5fcd71f789bdc7d53357bae16db21b1b54aedd5760e97a',
 'engine.py': '25a2ba5fda0c4ed06602d65ceef3a061f8bdf9c021740aa59fbde05198498765',
 'eval_utils.py': 'e38819dc53085912c128401c852e86382f2d52abf1cb0e028ff9a72c43bbf9d8',
 'model.py': 'c629f59de63f4d6e3b3554a3a2b3c425b504ec4b92134167b60407b3dbbac1e8'}


def env_path(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError("Set " + name + "; see README.md")
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def model_source():
    root = env_path("GARQ_MODEL_SOURCE", PACKAGE_ROOT.parent / "scalability/source_server")
    for name, expected in BACKEND_HASHES.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Required GARQ source is missing or changed: " + str(path))
    return root


def read_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() != ".py":
        return json.loads(text)
    tree = ast.parse(text, filename=str(path))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "CONFIG"]
    if len(assignments) != 1:
        raise ValueError("Python configurations require one literal CONFIG assignment")
    return ast.literal_eval(assignments[0].value)


def relocate(value):
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if not isinstance(value, str):
        return value
    value = value.replace("${GARQ_MODULE_ROOT}", str(PACKAGE_ROOT))
    if "${GARQ_BACKEND_ROOT}" in value:
        value = value.replace("${GARQ_BACKEND_ROOT}", str(model_source()))
    value = os.path.expandvars(value)
    filename = os.environ.get("GARQ_INPUT_PATH_MAP")
    if filename:
        prefixes = read_config(filename)
        for prefix in sorted(prefixes, key=len, reverse=True):
            if value == prefix or value.startswith(prefix.rstrip("/\\") + "/") or value.startswith(prefix.rstrip("/\\") + "\\"):
                value = str(prefixes[prefix]) + value[len(prefix):]
                break
    if re.search(r"\$\{[^}]+\}", value):
        raise RuntimeError("Unset input path variable: " + value)
    return value


def load_json(path):
    """Read a Python configuration or an existing JSON run record.

    Existing JSON files are read without rewriting their bytes, preserving
    external manifest and assignment SHA checks in the analysis scripts.
    """
    return relocate(read_config(path))
