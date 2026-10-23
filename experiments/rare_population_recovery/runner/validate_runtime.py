"""Import/source checks and original quantizer toy equivalence, without fitting data."""
import argparse
import json
from pathlib import Path
from common import read_json, validate_registry, import_exact, runtime_versions, source_record, write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--method',choices=['prepare','GARQ_quantizer','SEACells','KMeans'],required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    registry=validate_registry(read_json(args.registry))
    result={'method':args.method,'source_hashes':source_record(registry),'training_performed':False}
    if args.method=='prepare':
        module=import_exact('r69_preflight_standard',registry['sources']['representation'])
        assert (module.N_HVG,module.N_PCA_RNA,module.N_LSI_ATAC,module.N_PCA_ADT,module.RANDOM_STATE)==(3000,50,50,20,0)
        from sklearn.decomposition import PCA
    elif args.method=='GARQ_quantizer':
        import torch,faiss
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        module=import_exact('r69_preflight_quantizer',registry['sources']['benchmark'])
        module.BASE=Path(registry['sources']['model']['path']).parent
        result['original_equivalence_test']=module.check_quantizer_equivalence()
        result['faiss_version']=getattr(faiss,'__version__','unknown')
        result['cuda']=torch.version.cuda
    elif args.method=='SEACells':
        import SEACells,cupy
        if cupy.cuda.runtime.getDeviceCount()<1:
            raise RuntimeError('CuPy cannot see the selected CUDA device')
        result['SEACells_import']=str(SEACells.__file__)
        result['cupy_import']=str(cupy.__file__)
    else:
        from sklearn.cluster import KMeans
    metric=import_exact('r69_preflight_metrics',registry['sources']['metrics'])
    result['metrics_test']=metric.test_metrics()
    result.update(status='PASS',versions=runtime_versions())
    if args.output:
        write_json(args.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
