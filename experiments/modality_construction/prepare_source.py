"""Copy the verified model and prepare the training-through-inference module."""
import argparse
from pathlib import Path
from release_paths import model_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    # This existing function copies source and extracts its training prefix;
    # it does not construct a model, load data or execute training.
    from r2c1 import snapshot_code
    snapshot_code({"code_dir": str(model_source())}, args.root.resolve())
    print(args.root.resolve() / "source")


if __name__ == "__main__":
    main()
