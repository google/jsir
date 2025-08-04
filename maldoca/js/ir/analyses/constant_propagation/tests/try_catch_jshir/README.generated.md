To run manually:

```shell
blaze run //third_party/maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/third_party/maldoca/js/ir/analyses/constant_propagation/tests/try_catch_jshir/input.js \
  --passes "source2ast,ast2hir" \
  --jsir_analysis constant_propagation
```
