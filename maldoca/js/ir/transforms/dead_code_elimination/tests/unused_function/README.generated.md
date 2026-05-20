To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/dead_code_elimination/tests/unused_function/input.js \
  --passes "source2ast,ast2hir,dead_code_elimination,hir2ast,ast2source"
```
