# Final controlled representation comparison

This is the final D5/D11 primary compression-30 comparison. `config.py` retains all 30 conditions: standard with GARQ quantizer/SEACells/KMeans, learned or MOFA+ with GARQ quantizer, three aggregation seeds. Config readers also accept original external JSON files. Runtime outputs remain JSON.

```bash
export GARQ_CONTROL_INPUT=/your/controlled_inputs
export GARQ_CONTROL_OUTPUT=/your/new/controlled_results
python r2c2_benchmark.py --dataset D5 --representation standard --method GARQ_quantizer --target-k 403 --seed 0
python evaluation/evaluate_rerun.py --root "$GARQ_CONTROL_OUTPUT" --input-root "$GARQ_CONTROL_INPUT" --config config.py
```

The first command is one condition; evaluate the complete declared grid. The evaluator needs `inputs/{D5,D11}/{standard,MOFA}.npz`, `learned_training/{D5,D11}_seed1/embedding.npz`, the archived post-hoc sampling manifest/sample files and geometry caches. Copy `config.py` to the output root when using `validate_evaluation.py` without an explicit configuration option.

The benchmark reuses byte-identical installed public `model.py` after a SHA256 check. Standard representation functions and metrics remain supplied. Upstream input generation is in `../representation_preparation`; its historical 60-run evaluator is separate from this final evaluator. Do not relabel the D1-D12 shared-embedding table as an independently refit GARQ comparison.
