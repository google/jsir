To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/dead_code_elimination/tests/while_statement/input.js \
  --passes "source2ast,extract_prelude,erase_comments,ast2hir,dynconstprop,dead_code_elimination,hir2ast,ast2source"
```
