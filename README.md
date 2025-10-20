# JSIR - Next Generation JavaScript Analysis Tooling

JSIR is a next-generation JavaScript analysis tool. At its core is an
[MLIR](https://mlir.llvm.org)-based high-level
[intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation),
which supports both
[dataflow analysis](https://en.wikipedia.org/wiki/Data-flow_analysis) and
lossless conversion back to source. This unique design makes it suitable for
source-to-source transformation.

## Use cases at Google

JSIR is used at Google for analyzing and detecting malicious JavaScript files,
protecting products like Ads, Android, and Chrome. Some example use cases are:

*   **Signal extraction**

    JSIR is used for extracting syntactical and behavioral signals, which are
    fed into downstream security systems.

*   **[Taint analysis](https://en.wikipedia.org/wiki/Taint_checking)**

    JSIR is used for detecting suspicious information flows, by utilizing its
    dataflow analysis capability.

*   **Decompilation**

    JSIR is used for decompiling the
    [Hermes](https://github.com/facebook/hermes) bytecode all the way to
    JavaScript code, by utilizing its ability to be fully lifted back to source
    code.

*   **Deobfuscation**:

    JSIR is used for deobfuscating JavaScript by utilizing its source-to-source
    transformation capability.

    See our latest [paper](https://arxiv.org/abs/2507.17691) on how we combine
    the Gemini LLM and JSIR for deobfuscation.

## Design highlights

Driven by the diverse use cases of malicious JavaScript analysis and detection,
JSIR needs to achieve two seemingly conflicting goals:

*   It needs to be **high-level** enough to be lifted back to the AST, in order
    to support source-to-source transformation and decompilation.

*   It needs to be **low-level** enough to facilitate dataflow analysis, in
    order to support taint analysis, constant propagation, etc..

To achieve these goals, JSIR defines a high level IR that uses MLIR regions to accurately model control flow structures.

See
[intermediate_representation_design.md](docs/intermediate_representation_design.md)
for details.

## Getting started

### Docker (Recommended)

The easiest way to get started with JSIR is using Docker:

```shell
# Build the Docker image
docker build -t jsir:latest .

# Run jsir_gen
docker run --rm jsir:latest jsir_gen --help

# Analyze a JavaScript file
docker run --rm -v $(pwd):/workspace jsir:latest jsir_gen --input_file=/workspace/yourfile.js
```


### Install build tools

We have only tested `clang` on Linux:

```shell
# Install clang:

sudo apt update
sudo apt install clang
```

We use the `Bazel` build system. It is recommended to use `Bazelisk` to manage
`Bazel` versions:

```shell
# Install Bazelisk through npm:

sudo apt install npm
sudo npm install -g @bazel/bazelisk
```

### Build

Note: The build takes a lot of storage space. If you run out of space, Bazel
will return a cryptic error.

LLVM takes a long time to fetch and build. We can test if LLVM is properly
included by building a part of it:

```shell
# This will fetch LLVM and build its support library:

bazelisk build @llvm-project//llvm:Support
```

To build JSIR:

```shell
# Build everything:
bazelisk build //...

# Or, build a single target:
bazelisk build //maldoca/js/ir:jsir_gen

# Or, build all targets in a directory:
bazelisk build //maldoca/js/ir/...
```

### Test

To run test cases:

```shell
# Run all tests:
bazelisk test //...

# Or, run a specific test:
bazelisk test //maldoca/js/quickjs:quickjs_test

# Or, run all tests under a directory:
bazelisk test //maldoca/js/ir/conversion/...
```

### Run the `jsir_gen` tool

Convert a JavaScript source file to JSHIR:

```shell
bazelisk run //maldoca/js/ir:jsir_gen --\
    --input_file=$(pwd)/maldoca/js/ir/conversion/tests/if_statement/input.js \
    --passes=source2ast,ast2hir
```

## Other links

*   **Adversarial JavaScript Analysis with MLIR**

    Talk at LLVM Developers' Meeting 2024

    [YouTube](https://www.youtube.com/watch?v=SY1ft5EXI3I)
    [Slides](https://llvm.org/devmtg/2024-10/slides/techtalk/Tan-JSIR.pdf)

*   **CASCADE: LLM-Powered JavaScript Deobfuscator at Google**

    Paper about combining LLM + JSIR for JavaScript deobfuscation

    [arXiv](https://arxiv.org/abs/2507.17691)

## DISCLAIMER

This is not an official Google product.
