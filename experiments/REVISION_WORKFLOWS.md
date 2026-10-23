# Revision analysis workflows

The original model and tutorial are at the repository root. These workflows use
their recorded model versions and input identities; they do not replace the
original implementation. Keep matrices, saved assignments, trajectories and
checkpoints outside the source tree.

| Reviewer topic | Workflow | Main purpose |
|---|---|---|
| R2Q1 | [Modality construction](modality_construction/) | D17/D18 modality construction, count-thinning and saved-assignment evaluation |
| R2Q1–Q2 | [Revision analyses](comparative_analysis/) | Common-reference information retention and three-arm quantizer controls |
| R2Q2 | [Representation inputs](representation_preparation/) and [primary comparison](representation_benchmark/) | D5/D11 standard, learned and MOFA+ inputs; aggregation comparisons at the recorded resolutions |
| R2Q2 | [Abundance reduction](rare_population_recovery/) | Reduce population abundance before reconstructing the representation and fitting aggregators |
| R2Q3 | [Kidney trajectory metrics](trajectory_metrics/) and [revision analyses](comparative_analysis/) | Shared-reference ordering/program metrics, 200-gene panel and root-only trajectory sensitivity |
| R2Q4 | [Matched analyses](matched_association/), [D18 extensions](downstream_extensions/) and [revision analyses](comparative_analysis/) | Matched downstream comparisons, RNA–protein analyses and equal-link-budget comparisons |
| R2Q5 | [Computational profiling](scalability/), [repeated training](training_reproducibility/) and [revision analyses](comparative_analysis/) | Runtime/memory, training and inference batch controls, saved-assignment and inherited-label evaluation |

Each linked README gives the entry points and required external inputs. The
primary controlled evaluator uses its 30-condition primary-resolution grid;
the original dual-resolution evaluator is retained separately. Count thinning,
modality down-weighting, frozen inference and repeated training are distinct
analyses. Their settings should not be substituted for one another.

## Example

With the original inputs at the locations declared in the settings:

```bash
export GARQ_INPUT_ROOT=/path/to/revision-inputs
export GARQ_REVISION_OUTPUT=/path/to/new-output
python experiments/comparative_analysis/code/module_B_v2.py
```

New configuration files are Python literal dictionaries or R lists. Readers
retain support for original external JSON input records. Runtime output formats
and input identity checks are unchanged. Use a fresh output directory; do not
replace a hash-bound input with another run merely because its filename or K
looks similar.

See [DEPENDENCIES.md](DEPENDENCIES.md) for the separate Python/R environments.
The code bundle excludes raw matrices, checkpoints and saved analysis inputs.
Some historical figure-generation and input-preparation steps are not covered
by these revision workflows; the workflow READMEs specify their input boundary.
