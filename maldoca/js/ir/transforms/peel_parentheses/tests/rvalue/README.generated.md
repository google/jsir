To run manually:

```shell
bazel run //maldoca/js/ir:jsir_gen -- \
  --input_file $(pwd)/maldoca/js/ir/transforms/peel_parentheses/tests/rvalue/input.js \
  --passes "source2ast,erase_comments,ast2jsir,peelparens,jsir2ast,ast2source"
```
