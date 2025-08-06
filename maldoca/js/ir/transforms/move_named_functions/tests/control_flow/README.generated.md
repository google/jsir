To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/move_named_functions/tests/control_flow/input.js \
  --passes "source2ast,erase_comments,ast2hir,movenamedfuncs,hir2ast,ast2source"
```
