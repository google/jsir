To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/dynamic_constant_propagation/tests/function_inline_second/input.js \
  --passes "source2ast,extract_prelude,erase_comments,ast2hir,dynconstprop,hir2ast,ast2source"
```
