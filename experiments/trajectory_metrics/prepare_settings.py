"""Prepare a fresh kidney analysis directory from literal settings."""
import argparse
import ast
import json
from pathlib import Path


def read_settings(path):
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    for key in ("ANALYSIS_PLAN", "ANALYSIS_AMENDMENT"):
        if key not in values or not isinstance(values[key], dict):
            raise ValueError("Missing dictionary setting: " + key)
    return values


def relocate(value, input_root):
    if isinstance(value, dict):
        return {k: relocate(v, input_root) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, input_root) for v in value]
    if isinstance(value, str) and value.startswith("/path/to/project_home/") and input_root:
        return str(Path(input_root) / value[len("/path/to/project_home/"):])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path(__file__).with_name("settings.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Replacement for /path/to/project_home")
    args = parser.parse_args()
    values = read_settings(args.settings)
    outputs = {"analysis_plan.json": values["ANALYSIS_PLAN"],
               "analysis_amendment_before_results.json": values["ANALYSIS_AMENDMENT"]}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty analysis directory: " + str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (args.output / name).write_text(json.dumps(relocate(value, args.input_root), indent=2, allow_nan=False), encoding="utf8")
    print("Prepared analysis settings in", args.output)


if __name__ == "__main__":
    main()
