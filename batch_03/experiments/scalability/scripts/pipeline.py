#!/usr/bin/env python
import os,sys,time,json,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
lock=(ROOT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
baselines=['Ma_RNA_ATAC_K644_B512_seed1','GSE164378_RNA_ADT_K500_B512_seed1']
def state(stage,**kwargs):
 p=ROOT/'pipeline_state.json';tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps({'stage':stage,'pid':os.getpid(),'time':time.time(),**kwargs},indent=2));tmp.replace(p)
def execute(script,args,logname):
 state('executing',script=script,args=args)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['MPLBACKEND']='Agg'
 with (ROOT/logname).open('a') as f:
  subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 state('waiting_for_large_baselines')
 while True:
  for name in baselines:
   p=ROOT/'runs'/name
   if (p/'failed.json').exists():raise RuntimeError('Baseline failed '+name)
   if (p/'resource_summary.json').exists():
    result=json.loads((p/'resource_summary.json').read_text())
    if result['returncode']!=0:raise RuntimeError('Baseline process failed '+name)
  if all((ROOT/'runs'/x/'completed.json').exists() and (ROOT/'runs'/x/'resource_summary.json').exists() for x in baselines):break
  qs=json.loads((ROOT/'queue_state.json').read_text())
  if qs.get('status')=='failed':raise RuntimeError('Baseline supervisor failed '+str(qs))
  time.sleep(30)
 execute('build_deliverables.py',[],'build_deliverables.log')
 execute('frozen_stability.py',['SMOKE_instrumentation_v2'],'audit/frozen_smoke.log')
 for name in baselines:execute('frozen_stability.py',[name],name+'_frozen.log')
 execute('build_deliverables.py',[],'build_deliverables.log')
 configs=json.loads((ROOT/'configs/training_sensitivity_queue.json').read_text())
 gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 free={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu_info.strip().splitlines()}
 if free.get(0,0)>=16000 and free.get(1,0)>=16000:
  queues={0:[],1:[]}
  for name in configs:
   c=json.loads((ROOT/'configs'/name).read_text())
   gpu=1 if c['dataset']=='D5_RNA_ADT' else 0
   c['gpu']=gpu;(ROOT/'configs'/name).write_text(json.dumps(c,indent=2));queues[gpu].append(name)
  state('parallel_training_sensitivity',configuration_count=len(configs),gpu_assignment={'D5_RNA_ADT':1,'D11_RNA_ATAC':0})
  running=[]
  for gpu,names in queues.items():
   env=os.environ.copy();env['GARQ_QUEUE_STATE']=f'queue_training_gpu{gpu}.json'
   log=(ROOT/f'training_sensitivity_gpu{gpu}.log').open('a')
   p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/supervisor.py'),*names],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   running.append((gpu,p,log))
  codes=[]
  for gpu,p,log in running:
   codes.append((gpu,p.wait()));log.close()
  if any(code!=0 for gpu,code in codes):raise RuntimeError('Training sensitivity queue failures '+str(codes))
 else:
  for name in configs:
   c=json.loads((ROOT/'configs'/name).read_text());c['gpu']=0
   (ROOT/'configs'/name).write_text(json.dumps(c,indent=2))
  execute('supervisor.py',configs,'training_sensitivity_queue.log')
 execute('evaluate_training_batches.py',[],'training_fixed_evaluation.log')
 execute('build_deliverables.py',[],'build_deliverables.log')
 state('all_experiments_and_artifacts_complete',package=str(ROOT/'GARQ_Scalability_Revision_Server_Package.zip'),review_required='Inspect figures and final wording before manuscript submission.')
except BaseException as e:
 state('failed',error=repr(e),traceback=traceback.format_exc());raise
