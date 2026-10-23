# GARQ experiments

See [REVISION_WORKFLOWS.md](REVISION_WORKFLOWS.md) for the main workflow index,
[DEPENDENCIES.md](DEPENDENCIES.md) for requirements and
[ENVIRONMENT_VERSIONS.md](ENVIRONMENT_VERSIONS.md) for recorded environments.

Existing entry points are retained:

- [scalability](scalability/): profiling and training/inference batch controls;
- [Benchmark controls](benchmark_controls/): resolution, modality, rare-population and related controls;
- [matched_association](matched_association/): matched downstream comparisons.

The additional workflows in the index use these public dependencies where their
source versions match. Saved inputs and result archives are not bundled with the
code. Each workflow specifies which files must be supplied.
