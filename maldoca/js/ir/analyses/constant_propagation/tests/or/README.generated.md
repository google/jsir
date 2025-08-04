To run manually:

```shell
blaze run //third_party/maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/third_party/maldoca/js/ir/analyses/constant_propagation/tests/or/input.js \
  --passes "source2ast,ast2hir,hir2lir" \
  --jsir_analysis constant_propagation
```
