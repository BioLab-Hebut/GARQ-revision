# Selected GARQ revision workflows

This code-only package preserves selected existing revision workflows and records their source hashes. Release preparation did not fit models, execute notebooks/R or generate experimental results. The public baseline at repository root is separate from the archived instrumented backends here.

- `scalability/`: final R2Q5/R1Q3 full-process profiling, frozen inference batch/order diagnostics, and the training batch-size grid. Its five-module server backend is independent.
- `phase2/`: selected earlier modality, count-thinning/permutation, training/inference sensitivity and D13/D16 profiling configurations, with their instrumented backend.
- `matched_association/`: archived matched-membership downstream analyses and later broad Signac/GO analysis code. Source labels distinguish archived execution copies from later additions.

`source_manifest.json` records original and published hashes and all preparation edits. Server usernames were replaced with neutral path prefixes; numerical settings were preserved. New launchers are release helpers, not original experiment code.

No raw or clinical data, saved model/checkpoints, expression arrays, membership tables, logs or SupplementaryData1 are included. Each workflow requires user-provided inputs. This package does not assert that all manuscript experiments or unified D13-D16 multibatch MOFA integration are included. A missing local source file does not establish whether an experiment was performed.

Validation in this release is static Python compilation and layout/provenance checks only. See each workflow README for the existing protocol and its required inputs.
