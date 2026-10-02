"""Prepare or run selected existing scalability workflows in a separate output root."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASELINES=['Ma_RNA_ATAC_K644_B512_seed1','GSE164378_RNA_ADT_K500_B512_seed1']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--stage',choices=['prepare','profiles','frozen','training','summary','all'],default='prepare')
    args=parser.parse_args()
    work=args.output_root.resolve()
    if work==ROOT or ROOT in work.parents:
        parser.error('--output-root must be outside the source package')
    work.mkdir(parents=True,exist_ok=True)
    for folder in ['scripts','source_server']:
        shutil.copytree(ROOT/folder,work/folder,dirs_exist_ok=True)
    for folder in ['configs','audit','runs','deliverables']:
        (work/folder).mkdir(exist_ok=True)
    for src in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(src.read_text(encoding='utf-8-sig'))
        if isinstance(cfg,dict):
            cfg['data_files']=[str(args.data_root.resolve()/p.removeprefix('/workspace/garq/data/')) for p in cfg['data_files']]
            cfg['gpu']=args.gpu
        (work/'configs'/src.name).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    shutil.copy2(ROOT/'README_ZH.md',work/'README_ZH.md')
    (work/'release_launch.json').write_text(json.dumps({'stage':args.stage,'gpu':args.gpu,'data_root':str(args.data_root.resolve()),'source_directory':str(ROOT),'numerical_settings_changed':False},indent=2),encoding='utf-8')
    def run(script,*arguments):
        subprocess.run([sys.executable,str(work/'scripts'/script),*arguments],cwd=work,check=True)
    print(json.dumps({'prepared':str(work),'stage':args.stage}))
    if args.stage=='prepare':return
    if args.stage in ['profiles','all']:
        run('supervisor.py',*[name+'.json' for name in BASELINES])
    if args.stage in ['frozen','all']:
        for name in BASELINES:run('frozen_stability.py',name)
    if args.stage in ['training','all']:
        names=json.loads((work/'configs/training_sensitivity_queue.json').read_text())
        run('supervisor.py',*names)
        run('evaluate_training_batches.py')
    if args.stage in ['summary','all']:
        run('build_deliverables.py')

if __name__=='__main__':main()
