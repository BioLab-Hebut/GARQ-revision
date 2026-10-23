from release_paths import PACKAGE_ROOT, env_path, load_json
from pathlib import Path
import argparse,json,os,subprocess,time,sys
ROOT=env_path('GARQ_R2C2_ROOT',PACKAGE_ROOT/'results')
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--gpu',required=True);args=p.parse_args()
(ROOT/"logs").mkdir(parents=True,exist_ok=True)
ds=args.dataset
env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
prereq=[ROOT/'learned_training'/f'{ds}_seed1'/'DONE.json',ROOT/'inputs'/ds/'MOFA.npz']
start=time.time()
while not all(x.exists() for x in prereq):
 if time.time()-start>7200:raise RuntimeError('Upstream representation did not complete within two hours')
 time.sleep(20)
grid=[]
for k in ({'D5':[403,242],'D11':[321,193]}[ds]):
 for seed in [0,1,2]:
  for rep,method in [('standard','GARQ_quantizer'),('learned','GARQ_quantizer'),('MOFA','GARQ_quantizer'),('standard','KMeans'),('standard','SEACells')]:
   grid.append((k,seed,rep,method))
state=[]
for k,seed,rep,method in grid:
 name=f'{ds}__{rep}__{method}__K{k}__s{seed}'
 python=str(env_path('GARQ_SEACELLS_PYTHON' if method=='SEACells' else 'GARQ_TRAIN_PYTHON',sys.executable))
 command=[python,str(PACKAGE_ROOT/'r2c2_benchmark.py'),'--dataset',ds,'--representation',rep,'--method',method,'--target-k',str(k),'--seed',str(seed)]
 print('START',name,flush=True)
 with open(ROOT/'logs'/f'{name}.log','a') as log:
  completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 state.append({'job':name,'returncode':completed.returncode,'finished_at':time.time()})
 tmp=ROOT/f'controller_{ds}_state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(ROOT/f'controller_{ds}_state.json')
 print('END',name,completed.returncode,flush=True)
print('GRID_COMPLETE',ds,flush=True)
