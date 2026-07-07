To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/split_declaration_statements/tests/variable_declaration/input.js \
  --passes "source2ast,ast2jsir,split_declaration_statements,jsir2ast,ast2source"
```
