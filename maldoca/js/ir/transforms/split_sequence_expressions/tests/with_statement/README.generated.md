To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/split_sequence_expressions/tests/with_statement/input.js \
  --passes "source2ast,ast2hir,split_sequence_expressions,hir2ast,ast2source"
```
