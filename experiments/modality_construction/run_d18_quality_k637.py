"""Run a prespecified D18 K637 seed1 clean/thinning quartet on GPU1.

Only writes under --output. Historical prepared counts are immutable inputs.
Each condition is run once, with the same requested K; no realized-K matching,
seed search or result-based selection is performed. Run in the GARQ environment.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from release_paths import PACKAGE_ROOT, env_path, load_json, model_source
BASE = model_source()
NEW_ROOT = env_path('GARQ_MODALITY_OUTPUT',PACKAGE_ROOT/'results')
NEW_CODE = PACKAGE_ROOT
OLD_ROOT = env_path('GARQ_REFERENCE_ROOT',PACKAGE_ROOT/'external_references')
MODS = ('RNA', 'ATAC', 'ADT')
CONDITIONS = ('clean', 'RNA_thin50', 'ATAC_thin50', 'ADT_thin50')
N, K, SEED, P, PSEED = 25517, 637, 1, .5, 20260918


def require(value, message):
    if not value:
        raise RuntimeError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return load_json(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def unchanged_or_save(path, value):
    if path.exists():
        require(read(path)==value, 'Existing settings differ: '+str(path))
    else:
        save(path, value)


def gpu_memory(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True)
    free, total = [int(x.strip()) for x in text.strip().split(',')]
    return {'gpu':gpu, 'free_MiB':free, 'total_MiB':total, 'at':now()}


def check_source(source, base):
    manifest = read(source/'source_manifest.json')
    for filename, digest in manifest['sha256'].items():
        require(sha(source/filename)==digest, 'Snapshot source changed: '+filename)
        require(sha(base/filename)==digest, 'Live source differs from frozen snapshot: '+filename)
    require(sha(source/'training_through_inference.py')==manifest['transformed_sha256'],
            'Transformed training module does not match source manifest')
    return sha(source/'source_manifest.json')


def prepare(args):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    sys.path.insert(0, str(args.code))
    from r2c1_core import ids_hash, thin_count_values

    old = args.old_root/'prepared/D18'
    local = args.output/'prepared'
    local.mkdir(parents=True, exist_ok=True)
    old_manifest = read(old/'manifest.json')
    require(old_manifest['n_cells']==N and SEED in old_manifest['seeds'], 'Unexpected original D18 manifest')
    require(old_manifest['thinning_probability']==P and old_manifest['perturbation_seed_base']+SEED==PSEED,
            'Existing ATAC thinning does not follow the specified seed/probability')
    cells = pd.read_csv(old/'cells.csv', dtype=str)
    require(len(cells)==N and cells.cell_id.is_unique and not cells.celltype.isna().any(), 'Invalid fixed D18 cells')
    require(ids_hash(cells.cell_id)==old_manifest['cell_ids_sha256'], 'Cell ID hash mismatch')
    settings = {'dataset':'D18','n_cells':N,'requested_k':K,'seed':SEED,
        'thinning_probability':P,'thinning_random_seed':PSEED,
        'thinning_rule':'numpy.default_rng(20260917 + training_seed).binomial(count,0.5), independently for each perturbed modality',
        'old_prepared_manifest':str(old/'manifest.json'),'old_manifest_sha256':sha(old/'manifest.json'),
        'cells_csv':str(old/'cells.csv'),'cells_csv_sha256':sha(old/'cells.csv'),
        'cell_ids_sha256':ids_hash(cells.cell_id)}
    unchanged_or_save(local/'settings.json', settings)
    all_paths, records = {}, {}
    for mod in MODS:
        clean = old/(mod+'_clean.h5ad')
        clean_digest = sha(clean)
        require(clean_digest==old_manifest['prepared_sha256'][clean.name], 'Old clean prepared input changed: '+mod)
        all_paths[mod+'_clean'] = str(clean)
        a = ad.read_h5ad(clean)
        require(a.n_obs==N and np.array_equal(a.obs_names.to_numpy(), cells.cell_id.to_numpy()), 'Prepared row mismatch: '+mod)
        require(np.array_equal(a.obs.celltype.astype(str).to_numpy(), cells.celltype.to_numpy()), 'Prepared labels mismatch: '+mod)
        x = sparse.csr_matrix(a.X)
        require(np.isfinite(x.data).all() and (x.data>=0).all() and np.allclose(x.data,np.rint(x.data),atol=1e-6,rtol=0),
                'Prepared X is not nonnegative integer counts: '+mod)
        require((np.asarray(x.sum(axis=1)).ravel()>0).all(), 'Zero-count clean cells: '+mod)
        thin_path = old/f'ATAC_thin50_seed{SEED}.h5ad' if mod=='ATAC' else local/f'{mod}_thin50_seed{SEED}.h5ad'
        if mod=='ATAC':
            require(sha(thin_path)==old_manifest['prepared_sha256'][thin_path.name], 'Existing ATAC thinning changed')
        elif not thin_path.exists():
            thin = x.copy()
            thin.data = thin_count_values(x.data, P, PSEED).astype(x.dtype)
            thin.eliminate_zeros()
            require((np.asarray(thin.sum(axis=1)).ravel()>0).all(), 'Thinning creates zero-count cells: '+mod)
            temp = local/(thin_path.stem+'.partial.h5ad')
            ad.AnnData(X=thin,obs=a.obs.copy(),var=a.var.copy()).write_h5ad(temp, compression='gzip')
            temp.replace(thin_path)
        thinned = ad.read_h5ad(thin_path)
        require(thinned.shape==a.shape and np.array_equal(thinned.obs_names.to_numpy(),a.obs_names.to_numpy())
                and np.array_equal(thinned.var_names.to_numpy(),a.var_names.to_numpy()), 'Thinned rows/features mismatch: '+mod)
        require(np.array_equal(thinned.obs.celltype.astype(str).to_numpy(),cells.celltype.to_numpy()), 'Thinned labels mismatch: '+mod)
        tx = sparse.csr_matrix(thinned.X)
        require(np.isfinite(tx.data).all() and (tx.data>=0).all() and np.allclose(tx.data,np.rint(tx.data),atol=1e-6,rtol=0),
                'Thinned counts invalid: '+mod)
        require((np.asarray(tx.sum(axis=1)).ravel()>0).all(), 'Thinned zero-count cells: '+mod)
        # A deterministic equality check independently verifies both reused and newly generated thinning.
        expected = x.copy()
        expected.data = thin_count_values(x.data,P,PSEED).astype(x.dtype)
        expected.eliminate_zeros()
        require((tx!=expected).nnz==0, 'Thinning values do not match fixed seed rule: '+mod)
        all_paths[mod+'_thin50'] = str(thin_path)
        records[mod] = {'clean_path':str(clean),'clean_sha256':clean_digest,'thin_path':str(thin_path),
            'thin_sha256':sha(thin_path),'n_cells':N,'n_features':int(a.n_vars),
            'clean_count_sum':float(x.sum()),'thinned_count_sum':float(tx.sum()),
            'no_zero_count_cells':True,'verified_exact_binomial_counts':True}
        del a,x,thinned,tx,expected
    manifest = {**settings, 'modalities':records,'data_paths':all_paths,
        'historical_inputs_modified':False,'clean_inputs_reused':True}
    unchanged_or_save(local/'manifest.json', manifest)
    return manifest


def make_jobs(args, prepared, source_hash):
    jobs = []
    for condition in CONDITIONS:
        name = 'D18_RNA_ATAC_ADT_'+condition+'_seed1_K637'
        run = args.output/'runs'/name
        run.mkdir(parents=True, exist_ok=True)
        paths = [prepared['data_paths'][mod+('_thin50' if condition==mod+'_thin50' else '_clean')] for mod in MODS]
        job = {'name':name,'dataset':'D18','seed':SEED,'modalities':list(MODS),'condition':condition,
            'requested_k':K,'target_k':K,'epoch':300,'batch_size':512,'k_knn':5,'converge_threshold':10,
            'source_dir':str(args.source),'source_manifest_sha256':source_hash,
            'prepared_manifest_sha256':sha(args.output/'prepared/manifest.json'),
            'cells_csv':prepared['cells_csv'],'data_files':paths,
            'series':'new_K637_seed1_paired_quality','selection_rule':'Single prespecified run per condition; requested K637 fixed, record actual occupied K; no result selection'}
        unchanged_or_save(run/'job.json', job)
        jobs.append(run)
    return jobs


def check_finished(run):
    result = read(run/'result.json')
    require(result['job_sha256']==sha(run/'job.json'), 'Completed job provenance differs: '+str(run))
    require(result['n_cells']==N and result['requested_k']==K and 0<result['realized_k']<=K, 'Invalid completed resolution')
    require((run/'assignments.csv.gz').is_file(), 'Completed run lacks assignments')
    return result


def run(args):
    require(args.gpu>=0, 'GPU index must be nonnegative')
    require(1<=args.max_parallel<=2, 'Maximum two simultaneous D18 jobs')
    args.output.mkdir(parents=True, exist_ok=True)
    for source in (args.old_root,args.base/'save',args.source,args.code):
        require(args.output!=source and source not in args.output.parents, 'Output overlaps a protected source path')
    # flock releases on process exit; the file persists for readable PID/history.
    with (args.output/'controller.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another D18 quality controller holds controller.lock')
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())+'\n'); lock.flush()
        active = []
        for pidfile in (args.output/'runs').glob('*/pid.json') if (args.output/'runs').exists() else []:
            record = read(pidfile)
            command = Path('/proc')/str(record['pid'])/'cmdline'
            if command.is_file() and str(pidfile.parent/'job.json').encode() in command.read_bytes():
                active.append(record)
        require(not active, 'Existing workers are still active; do not duplicate launches: '+str(active))
        memory = gpu_memory(args.gpu)
        require(memory['free_MiB']>=16*1024, 'At least 16 GiB GPU1 free memory is required at startup: '+str(memory))
        save(args.output/'gpu_preflight.json',memory)
        source_hash = check_source(args.source,args.base)
        prepared = prepare(args)
        jobs = make_jobs(args,prepared,source_hash)
        save(args.output/'protocol.json',{'conditions':list(CONDITIONS),'seed':SEED,'requested_K':K,
            'n_paired_seeds':1,'probability':P,'thinning_seed':PSEED,'max_parallel':args.max_parallel,
            'GPU':args.gpu,'CPU_threads_per_worker':4,'clean_baseline':'new clean job in this quartet',
            'realized_K_policy':'Observed occupied counts reported without matching retries',
            'started_at':now(),'controller_pid':os.getpid(),'controller_sha256':sha(__file__)})
        pending = []
        for jobdir in jobs:
            if (jobdir/'result.json').exists():
                check_finished(jobdir)
                print('RESUME completed',jobdir.name,flush=True)
            else:
                pending.append(jobdir)
        running,failures = [],[]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),MPLBACKEND='Agg',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
        try:
            while pending or running:
                for item in running[:]:
                    process,jobdir,log = item
                    code = process.poll()
                    if code is not None:
                        log.close(); running.remove(item)
                        if code==0 and (jobdir/'result.json').exists():
                            result = check_finished(jobdir)
                            print('DONE',jobdir.name,'occupied_K',result['realized_k'],flush=True)
                        else:
                            failures.append({'run':jobdir.name,'exit_code':code})
                            save(jobdir/'failed.json',{'exit_code':code,'at':now()})
                while pending and len(running)<args.max_parallel:
                    memory = gpu_memory(args.gpu)
                    if memory['free_MiB']<6*1024:
                        break
                    require(check_source(args.source,args.base)==source_hash, 'Source changed since preflight')
                    jobdir = pending.pop(0)
                    log = (jobdir/'training.log').open('a',encoding='utf-8')
                    process = subprocess.Popen([sys.executable,str(args.code/'r2c1.py'),'worker','--job',str(jobdir/'job.json')],
                        env=env,stdout=log,stderr=subprocess.STDOUT,cwd=str(jobdir))
                    save(jobdir/'pid.json',{'pid':process.pid,'started_at':now(),'gpu_memory_before_launch':memory})
                    running.append((process,jobdir,log))
                    print('START',jobdir.name,'PID',process.pid,flush=True)
                    # Give the first worker a chance to allocate before checking the second launch.
                    if pending and len(running)<args.max_parallel:
                        time.sleep(20)
                save(args.output/'status.json',{'state':'running' if pending or running else ('failed' if failures else 'completed'),
                    'pending':[x.name for x in pending],'active':[{'run':x[1].name,'pid':x[0].pid} for x in running],
                    'failures':failures,'updated_at':now(),'controller_pid':os.getpid()})
                if pending or running:
                    time.sleep(15)
            require(not failures, 'One or more training conditions failed; inspect training.log; no automatic reselection')
            if not args.skip_evaluation:
                subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_d18_quality_k637.py')),
                    '--root',str(args.output),'--reference-root',str(args.old_root),
                    '--historical-evaluation-dir',str(args.historical_evaluation_dir)],check=True,env=env)
            save(args.output/'status.json',{'state':'completed','training_conditions':list(CONDITIONS),
                'evaluation_run':not args.skip_evaluation,'updated_at':now(),'controller_pid':os.getpid()})
        finally:
            # Do not terminate already launched scientific jobs on controller interruption.
            for _,_,log in running:
                log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--old-root',type=Path,default=OLD_ROOT)
    p.add_argument('--source',type=Path,default=NEW_ROOT/'source')
    p.add_argument('--code',type=Path,default=NEW_CODE)
    p.add_argument('--output',type=Path,default=NEW_ROOT/'D18_K637_quality')
    p.add_argument('--historical-evaluation-dir',type=Path,default=NEW_ROOT/'D18_K637_evaluation')
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--max-parallel',type=int,default=2)
    p.add_argument('--skip-evaluation',action='store_true')
    args = p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve())
    run(args)


if __name__=='__main__':
    main()
