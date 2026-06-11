"""Module extensions for non-BCR dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""

    # LLVM is pinned to the same tag used in the Google monorepo.
    # The build files from the LLVM monorepo are overlaid via llvm_configure
    # (called via use_repo_rule in MODULE.bazel) to produce @llvm-project.
    new_git_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        tag = "llvmorg-22-init",
        init_submodules = False,
        remote = "https://github.com/llvm/llvm-project.git",
    )

    # Optional LLVM dependencies for performance. The build has no way to omit
    # them. See https://reviews.llvm.org/D143344#4232172
    maybe(
        http_archive,
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

    http_archive(
        name = "quickjs",
        build_file = "@jsir//:bazel/quickjs.BUILD",
        sha256 = "3c4bf8f895bfa54beb486c8d1218112771ecfc5ac3be1036851ef41568212e03",
        urls = ["https://bellard.org/quickjs/quickjs-2024-01-13.tar.xz"],
        strip_prefix = "quickjs-2024-01-13",
        add_prefix = "quickjs",
    )

llvm_deps = module_extension(
    implementation = _llvm_deps_impl,
)
