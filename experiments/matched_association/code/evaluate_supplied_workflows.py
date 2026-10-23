"""Execute the supplied Pearson and glmnet workflows on each audited batch."""
import sys,time,json,os
from pathlib import Path
from run_bimodal_downstream import run_associations,run_tf
root=Path(__file__).resolve().parent.parent
tag=sys.argv[1];bd=root/'analysis'/tag
deadline=time.time()+3600
while not (bd/'MOFA_input_manifest.json').exists():
    if time.time()>deadline:raise TimeoutError('Aggregation manifest not ready')
    time.sleep(5)
arms=json.loads((bd/'MOFA_input_manifest.json').read_text())
assert len(arms)==(1 if tag=='single_cell' else 10)
print('Association analysis',tag,flush=True)
run_associations(bd,arms)
print('TF LASSO analysis',tag,flush=True)
run_tf(bd,arms,'/usr/bin/Rscript',1 if tag in ('full','single_cell') else int(tag[-1]))
(bd/'SUPPLIED_WORKFLOWS_DONE').write_text('Pearson RNA-ATAC / RNA-ADT and TF-gene glmnet executed successfully\n')
print('DONE',tag,flush=True)
