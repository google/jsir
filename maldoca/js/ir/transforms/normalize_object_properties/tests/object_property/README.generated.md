To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/normalize_object_properties/tests/object_property/input.js \
  --passes "source2ast,ast2jsir,normalizeobjprops,jsir2ast,ast2source"
```
