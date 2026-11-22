"""Module extensions for non-BCR dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""
    LLVM_COMMIT = "d5efcd0ec41d0f425076e3dd738a68d333ad610f"

    # LLVM is pinned to the same commit used in the Google monorepo.
    # The build files from the LLVM monorepo are overlaid via llvm_configure
    # (called via use_repo_rule in MODULE.bazel) to produce @llvm-project.
    new_git_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        commit = LLVM_COMMIT,
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )

    http_archive(
        name = "quickjs",
        build_file = "@jsir//:bazel/quickjs.BUILD",
        sha256 = "3c4bf8f895bfa54beb486c8d1218112771ecfc5ac3be1036851ef41568212e03",
        urls = ["https://bellard.org/quickjs/quickjs-2024-01-13.tar.xz"],
        strip_prefix = "quickjs-2024-01-13",
        add_prefix = "quickjs",
    )

    # The editor of the in-browser demo (//maldoca/js/ir/wasm), from npm.
    http_archive(
        name = "monaco_editor",
        build_file = "@jsir//:bazel/monaco_editor.BUILD",
        sha256 = "5dbda58750b13452de39affcea703af096022d8e3546c67ed5d3bd072763f939",
        urls = ["https://registry.npmjs.org/monaco-editor/-/monaco-editor-0.52.0.tgz"],
    )

llvm_deps = module_extension(
    implementation = _llvm_deps_impl,
)
