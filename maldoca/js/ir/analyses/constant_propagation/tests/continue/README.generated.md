To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/analyses/constant_propagation/tests/continue/input.js \
  --passes "source2ast,ast2hir,hir2lir" \
  --jsir_analysis constant_propagation
```
