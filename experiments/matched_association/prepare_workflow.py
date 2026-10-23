"""Materialize an existing matched analysis with user-supplied path mappings."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path-map',required=True,type=Path,help='JSON mapping neutral path prefixes to actual local paths.')
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    mapping=json.loads(args.path_map.read_text(encoding='utf-8-sig'))
    if not isinstance(mapping,dict) or any(not isinstance(x,str) for pair in mapping.items() for x in pair):
        parser.error('--path-map must contain a JSON object of string keys and values')
    work=args.output_root.resolve()
    if work==ROOT or ROOT in work.parents:
        parser.error('--output-root must be outside the source package')
    mapping['/workspace/garq/vscode/GARQ20260905/reviewer2_comment4_20260918']=work.as_posix()
    mapping['/workspace/garq/vscode/GARQ20260905/reviewer2_comment4_20260917/analysis']=(work/'analysis').as_posix()
    mapping['/usr/bin/Rscript']=mapping.get('/usr/bin/Rscript','Rscript')
    for src in (ROOT/'code').rglob('*'):
        if not src.is_file():continue
        text=src.read_text(encoding='utf-8-sig')
        for old,new in sorted(mapping.items(),key=lambda x:len(x[0]),reverse=True):
            # Linux/R paths use forward slashes; no source is executed here.
            text=text.replace(old,Path(new).as_posix() if new!='Rscript' else new)
        dst=work/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(text,encoding='utf-8')
    (work/'audit').mkdir(exist_ok=True)
    for src in (work/'code/source/Signac_1.14.0').glob('*.R'):
        shutil.copy2(src,work/'audit'/src.name)
    print(json.dumps({'prepared_root':str(work),'executed':False,'next_step':'Supply required annotation/metadata inputs described in README.md, then run the selected existing code script.'}))

if __name__=='__main__':main()
