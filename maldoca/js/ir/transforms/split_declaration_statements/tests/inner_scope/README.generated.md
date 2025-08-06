To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/split_declaration_statements/tests/inner_scope/input.js \
  --passes "source2ast,ast2hir,split_declaration_statements,hir2ast,ast2source"
```
