"""Serial, resumable queue; no calibration, result ranking or CPU fallback."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from common import load_plan, read_json, write_json, latest_success, now

HERE = Path(__file__).resolve().parent

def resources(registry, job):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':',1)
        if key == 'MemAvailable':
            memory['available_ram_gib'] = int(value.strip().split()[0])/1024**2
    need_ram = registry.get('min_ram_gib', {}).get(job.get('method', 'prepare'), 32)
    result = {'checked_utc':now(), **memory, 'required_ram_gib':need_ram}
    ok = memory.get('available_ram_gib',0) >= need_ram
    blocked_pids = [int(pid) for pid in registry.get('wait_for_process_ids',[]) if (Path('/proc')/str(int(pid))).exists()]
    if blocked_pids:
        result['waiting_for_existing_process_ids'] = blocked_pids
        ok = False
    if job.get('method') in ('GARQ_quantizer','SEACells'):
        index = int(registry['gpu_index'])
        process = subprocess.run(['nvidia-smi','-i',str(index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],
                                 capture_output=True,text=True,timeout=20)
        if process.returncode:
            result.update(gpu_error=process.stderr.strip(), status='WAITING')
            return False, result
        free = float(process.stdout.strip())
        required = registry.get('min_gpu_free_mib',{}).get(job['method'],16384)
        result.update(gpu_index=index,free_gpu_mib=free,required_gpu_mib=required)
        ok = ok and free >= required
    result['status'] = 'PASS' if ok else 'WAITING'
    return ok, result

def existing_failure(job_dir):
    return any(read_json(p).get('status')=='FAILED' for p in Path(job_dir).glob('attempt_*/result.json'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-failed', action='store_true',help='Create new attempts for prior technical failures; preserve all failed records')
    parser.add_argument('--wait-resources', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--job-timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    registry = plan['registry']
    root = Path(registry['output_root'])
    completed = [j['job_id'] for j in plan['jobs'] if latest_success(root/'jobs'/j['job_id'],plan['plan_sha256'])]
    if args.dry_run:
        print({'status':'PASS','scientific_execution':False,'planned_jobs':len(plan['jobs']),
               'prepare_jobs':24,'fit_jobs':144,'completed':len(completed),
               'plan_sha256':plan['plan_sha256'],'cuda_visible_devices':str(registry['gpu_index'])})
        return 0
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'queue.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A queue for this output directory is already running')
    state = {'plan_sha256':plan['plan_sha256'],'started_utc':now(),'queue_pid':os.getpid(),
             'status':'RUNNING','jobs':[],'resume_flags':{'retry_failed':args.retry_failed}}
    executed = 0
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(registry['gpu_index']),PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
        env[name] = '8'
    if registry.get('pythonpath'):
        env['PYTHONPATH'] = os.pathsep.join(registry['pythonpath'])
    for job in plan['jobs']:
        directory = root/'jobs'/job['job_id']
        done = latest_success(directory,plan['plan_sha256'])
        if done:
            state['jobs'].append({'job_id':job['job_id'],'status':'SUCCEEDED','reused_current_plan':True})
            continue
        if existing_failure(directory) and not args.retry_failed:
            state['jobs'].append({'job_id':job['job_id'],'status':'FAILED','action':'preserved; retry requires explicit flag'})
            continue
        if job['kind']=='fit' and not latest_success(root/'jobs'/(job['cohort']+'__prepare'),plan['plan_sha256']):
            state['jobs'].append({'job_id':job['job_id'],'status':'BLOCKED','reason':'cohort preparation incomplete'})
            continue
        while True:
            available, snapshot = resources(registry,job)
            state.update(current_job=job['job_id'],resource_preflight=snapshot,
                         status='RUNNING' if available else 'WAITING_RESOURCES',updated_utc=now())
            write_json(root/'queue_status.json',state)
            if available:
                break
            print({'job_id':job['job_id'],'resource_preflight':snapshot},flush=True)
            if not args.wait_resources:
                return 2
            time.sleep(30)
        attempt = directory/('attempt_'+str(time.time_ns()))
        attempt.mkdir(parents=True,exist_ok=False)
        interpreter = registry['python']['prepare' if job['kind']=='prepare' else job['method']]
        command = [interpreter,'-u',str(HERE/'worker.py'),'--plan',str(args.plan.resolve()),
                   '--job-id',job['job_id'],'--attempt-dir',str(attempt)]
        write_json(attempt/'launch.json',{'command':command,'resources':snapshot,'started_utc':now(),
                                         'plan_sha256':plan['plan_sha256']})
        print('START',job['job_id'],flush=True)
        with (attempt/'worker.log').open('x',encoding='utf-8') as log:
            try:
                process = subprocess.run(command,env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,
                                         timeout=args.job_timeout_seconds)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = -999
        if (attempt/'result.json').exists():
            result = read_json(attempt/'result.json')
        else:
            result = {'status':'FAILED','plan_sha256':plan['plan_sha256'],'job':job,
                      'finished_utc':now(),'error':'worker exited without a result; inspect preserved log',
                      'returncode':returncode,'attempt_dir':str(attempt)}
            write_json(attempt/'result.json',result)
        if returncode==0 and result.get('status')=='SUCCEEDED':
            write_json(directory/'DONE.json',result)
            latest_success(directory,plan['plan_sha256'])
            status='SUCCEEDED'
        else:
            status='FAILED'
        state['jobs'].append({'job_id':job['job_id'],'status':status,'returncode':returncode,'attempt_dir':str(attempt)})
        state.update(updated_utc=now())
        write_json(root/'queue_status.json',state)
        print('END',job['job_id'],status,flush=True)
        executed += 1
        if args.max_jobs is not None and executed>=args.max_jobs:
            state.update(status='PARTIAL_STOP_REQUESTED',updated_utc=now())
            write_json(root/'queue_status.json',state)
            return 0
    all_success = all(latest_success(root/'jobs'/j['job_id'],plan['plan_sha256']) for j in plan['jobs'])
    state.update(status='COMPLETE' if all_success else 'COMPLETE_WITH_FAILURES',finished_utc=now())
    write_json(root/'queue_status.json',state)
    subprocess.run([registry['python']['prepare'],str(HERE/'summarize.py'),'--plan',str(args.plan.resolve())],check=True)
    return 0 if all_success else 1

if __name__=='__main__':
    raise SystemExit(main())
