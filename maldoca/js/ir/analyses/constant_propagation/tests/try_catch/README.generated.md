To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/analyses/constant_propagation/tests/try_catch/input.js \
  --passes "source2ast,ast2hir" \
  --jsir_analysis constant_propagation
```
