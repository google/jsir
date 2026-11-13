To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/constant_propagation/dynamic_tests/scoped_alias/input.js \
  --passes "source2ast,extract_prelude,erase_comments,ast2jsir,dynconstprop,jsir2ast,ast2source"
```
