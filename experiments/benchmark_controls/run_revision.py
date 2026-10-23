"""Portable launcher; selects the archived instrumented backend explicitly."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREFIX = '/workspace/garq/data/'

def remap(value, data_root):
    if isinstance(value, str) and value.startswith(PREFIX):
        return str(data_root / value[len(PREFIX):])
    if isinstance(value, list):
        return [remap(x, data_root) for x in value]
    if isinstance(value, dict):
        return {k: remap(v, data_root) for k,v in value.items()}
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Run an existing experiment after writing a resolved config; omit to prepare only.')
    args = parser.parse_args()
    import yaml
    source = args.config if args.config.is_absolute() else ROOT / args.config
    config = remap(yaml.safe_load(source.read_text(encoding='utf-8-sig')), args.data_root.resolve())
    output = args.output_root.resolve()
    output.mkdir(parents=True,exist_ok=True)
    config['result_root'] = str(output)
    resolved = output / 'release_configs' / source.name
    resolved.parent.mkdir(parents=True,exist_ok=True)
    resolved.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    print(json.dumps({'resolved_config':str(resolved),'implementation_tag':config.get('implementation_tag'),'execution_requested':args.execute}))
    if args.execute:
        sys.path.insert(0,str(ROOT))
        sys.path.insert(0,str(ROOT/'vendor/phase2_backend'))
        os.environ['GARQ_DATA_ROOT']=str(args.data_root.resolve())
        os.chdir(ROOT)
        from revision_exp.run import run
        run(resolved)

if __name__ == '__main__':
    main()
