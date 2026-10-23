# Fixed-batch repeated training

The full recorded grid is D1-D12 × seeds 1/2/3, training batch 512 and inference batch 2048. `registry.py` and `run_plan.py` contain the scientific settings and recorded reuse conditions. Edit paths in private copies; keep all conditions and source hashes. External JSON registries/plans remain accepted.

```bash
python runner/run_queue.py --registry registry.py --manifest run_plan.py --work-root /your/new/results --source-root /your/repository/experiments/scalability --gpu 1 --validate-only
python runner/summarize.py --registry registry.py --manifest run_plan.py --work-root /your/results --out /your/summary
```

The backend is the existing `experiments/scalability/source_server`. Six historical D5/D11 runs can only be reused with their recorded identity evidence and saved annotations. The registry's D9 memory gate is an operational startup estimate; the historical D9 retry used separate resource guards. Input counts and saved run artifacts are external. Runtime completion and metric files remain JSON.
