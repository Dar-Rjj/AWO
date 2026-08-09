# Third-party sources

`upstream.lock.yaml` is the source of truth for upstream repositories and paper versions.

Do not copy a moving upstream default branch into this repository. Fetch the exact commit, retain its license, and put any local changes in `patches/` or a compatibility layer. Large datasets and result archives belong under ignored data directories and are verified through `data/manifests/aflow-artifacts.json`.
