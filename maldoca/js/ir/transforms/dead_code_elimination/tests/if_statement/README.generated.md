To run manually:

```shell
blaze run //third_party/maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/third_party/maldoca/js/ir/transforms/dead_code_elimination/tests/if_statement/input.js \
  --passes "source2ast,extract_prelude,erase_comments,ast2hir,dynconstprop,dead_code_elimination,hir2ast,ast2source"
```
