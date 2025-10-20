To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/constant_propagation/tests/void_0/input.js \
  --passes "source2ast,ast2hir,constprop,hir2ast,ast2source"
```
