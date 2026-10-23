# Pre-construction rare-population recovery

The recorded design has 24 retained cohorts and 144 fits: D5/D11, retained fractions 1/0.5/0.25/0.1, compression 30/50, GARQ quantizer/SEACells/KMeans, seeds 0/1/2. Original <=2% populations are thinned together; non-target cells remain. The shared standard representation is rebuilt before fitting aggregators. This does not retrain the full encoder/decoder.

Configure a private copy of `registry_example.py`; preserve scientific settings and hashes. It reuses installed public `model.py` with an exact hash guard. `protocol_counts.py` retains all classes, including zero recovery. External JSON configurations and generated JSON plans/results remain supported.

```bash
python runner/build_plan.py --registry registry_example.py --output /your/new/plan.json --validate-only
python runner/build_plan.py --registry registry_example.py --output /your/new/plan.json
python runner/run_queue.py --plan /your/new/plan.json --wait-resources
python validation/verify_results.py --plan /your/new/plan.json --output-dir /your/validation --require-complete
python validation/analyze_verified_results.py --validation-dir /your/validation --protocol-counts protocol_counts.py --output-dir /your/aggregates
```

All fixed classes and methods remain in the analysis; K is round(N_retained/compression), with no outcome-dependent adjustment. Counts, interpreters and original source/input identities must be supplied and verified before execution.
