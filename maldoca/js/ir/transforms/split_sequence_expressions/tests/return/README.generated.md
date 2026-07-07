To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/split_sequence_expressions/tests/return/input.js \
  --passes "source2ast,ast2jsir,split_sequence_expressions,jsir2ast,ast2source"
```
