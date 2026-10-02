#!/usr/bin/env python
import sys,os,time,json,subprocess,csv,platform,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
def write(p,d):p.write_text(json.dumps(d,indent=2,default=str))
def run(config):
 cfg=json.loads(config.read_text());out=ROOT/'runs'/cfg['run_id'];out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():return 0
 if psutil.virtual_memory().available<cfg.get('minimum_available_ram_gib',230)*1024**3:
  raise RuntimeError('Insufficient available host RAM; queue stopped before launch')
 gpu=int(cfg.get('gpu',0))
 query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 free={int(a.split(',')[0]):int(a.split(',')[1]) for a in query.strip().splitlines()}
 if free[gpu]<cfg.get('minimum_free_gpu_mib',16000):raise RuntimeError('Selected GPU has insufficient free memory')
 env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'})
 # Retain existing thread defaults; report them rather than silently tune performance.
 write(out/'hardware.json',{'hostname':platform.node(),'platform':platform.platform(),'cpu_count':psutil.cpu_count(),'total_ram_bytes':psutil.virtual_memory().total,'gpu_visibility':str(gpu),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),'lscpu':subprocess.check_output(['lscpu'],text=True),'environment':{k:env.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES']}})
 start=time.perf_counter();log=(out/'console.log').open('w')
 p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/profile_runner.py'),str(config)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
 write(out/'pid.json',{'pid':p.pid,'started_unix':time.time(),'config':str(config)})
 peak={'main_rss_bytes':0,'tree_rss_sum_bytes':0,'tree_pss_sum_bytes':0,'process_gpu_bytes':0};phase_peaks={}
 with (out/'resource_samples.csv').open('w') as f:
  fields=['seconds','phase','main_rss_bytes','tree_rss_sum_bytes','tree_pss_sum_bytes','process_gpu_bytes','processes']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  while p.poll() is None:
   phase='startup'
   try:phase=json.loads((out/'state.json').read_text()).get('phase','unknown')
   except (FileNotFoundError,json.JSONDecodeError):pass
   main=0;rss=0;pss=0;pids=[]
   try:
    proc=psutil.Process(p.pid);processes=[proc]+proc.children(recursive=True)
    for child in processes:
     try:
      mem=child.memory_full_info();pids.append(child.pid);rss+=mem.rss;pss+=getattr(mem,'pss',0)
      if child.pid==p.pid:main=mem.rss
     except (psutil.NoSuchProcess,psutil.AccessDenied,ProcessLookupError):pass
   except psutil.NoSuchProcess:pass
   gpu_used=0
   try:
    info=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
    for line in info.strip().splitlines():
     parts=line.split(',')
     if len(parts)==2 and parts[0].strip().isdigit() and int(parts[0]) in pids and parts[1].strip().isdigit():gpu_used+=int(parts[1])*1024**2
   except (subprocess.SubprocessError,ValueError):pass
   vals={'main_rss_bytes':main,'tree_rss_sum_bytes':rss,'tree_pss_sum_bytes':pss,'process_gpu_bytes':gpu_used}
   writer.writerow({'seconds':round(time.perf_counter()-start,3),'phase':phase,**vals,'processes':len(pids)});f.flush()
   for k,v in vals.items():
    peak[k]=max(peak[k],v)
    phase_peaks.setdefault(phase,{x:0 for x in peak})
    phase_peaks[phase][k]=max(phase_peaks[phase][k],v)
   time.sleep(0.5)
 log.close();code=p.wait()
 write(out/'resource_summary.json',{'returncode':code,'process_wall_seconds':time.perf_counter()-start,'peaks':peak,'phase_peaks':phase_peaks,'definitions':{'tree_pss_sum_bytes':'Sum of proportional set size of the training process and all live descendants at each sample; avoids double counting shared pages. Sample interval is collection time plus 0.5 s.','tree_rss_sum_bytes':'Auxiliary only: shared worker pages can be counted repeatedly.','process_gpu_bytes':'nvidia-smi compute-process memory for the job PID tree, including non-PyTorch CUDA allocations; sampled.','main_rss_bytes':'Sampled RSS of main process; ru_maxrss is additionally reported by runner.'}})
 return code
configs=[ROOT/'configs'/x for x in sys.argv[1:]]
try:
 for c in configs:
  write(ROOT/os.environ.get('GARQ_QUEUE_STATE','queue_state.json'),{'status':'running','config':str(c),'time':time.time(),'supervisor_pid':os.getpid()})
  code=run(c)
  if code:raise RuntimeError('Job failed '+str(c)+' exit '+str(code))
 write(ROOT/os.environ.get('GARQ_QUEUE_STATE','queue_state.json'),{'status':'complete','time':time.time()})
except BaseException as e:
 write(ROOT/os.environ.get('GARQ_QUEUE_STATE','queue_state.json'),{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise
