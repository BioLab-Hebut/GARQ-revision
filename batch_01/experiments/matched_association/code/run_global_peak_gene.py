"""Reuse existing memberships and run the broad matched peak-gene controls."""
import os, pathlib, subprocess, time, json
r=pathlib.Path('/workspace/garq/vscode/GARQ20260905/reviewer2_comment4_20260918')
arms=['GARQ_RNA_ATAC_ADT','SEACells','MetaCellV2','SuperCell','GARQ_RNA','GARQ_ATAC','GARQ_ADT','GARQ_RNA_ATAC','GARQ_RNA_ADT','GARQ_ATAC_ADT']
jobs=[('matched_seed1',a) for a in arms]+[('single_cell','SingleCell')]
env=dict(os.environ,R_LIBS_USER='/workspace/garq/R/x86_64-pc-linux-gnu-library/4.4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',GARQ_PEAK_WORKERS='4')
live=[];results=[]
for batch,arm in jobs:
    out=r/'peak_gene'/batch/arm
    if (out/'DONE').exists():
        results.append(dict(batch=batch,arm=arm,status='complete'));continue
    out.mkdir(parents=True,exist_ok=True)
    log=open(out/'run.log','w')
    p=subprocess.Popen(['Rscript',str(r/'code/global_peak_gene.R'),batch,arm],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    live.append((p,log,batch,arm));time.sleep(2)
while live:
    remain=[]
    for p,log,batch,arm in live:
        rc=p.poll()
        if rc is None: remain.append((p,log,batch,arm))
        else:
            log.close();results.append(dict(batch=batch,arm=arm,status='complete' if rc==0 else 'failed',exit_code=rc))
    live=remain
    (r/'peak_gene_status.json').write_text(json.dumps(dict(completed=results,running=[dict(batch=b,arm=a,pid=p.pid) for p,_,b,a in live]),indent=2))
    if live:time.sleep(30)
print(json.dumps(results),flush=True)
