"""Module extensions for non-BCR dependencies."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _llvm_deps_impl(_):
    """Implementation of the llvm_deps module extension."""
    LLVM_COMMIT = "030e74c2808a9af58c6b4ef461fd0c2c7039d647"

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

    esbuild_prebuilt(name = "esbuild")

llvm_deps = module_extension(
    implementation = _llvm_deps_impl,
)

# esbuild compiles and bundles the TypeScript Babel glue
# (//maldoca/js/babel_ts) into a single script for QuickJS. It is a
# self-contained Go binary, published to npm per platform. Keep the version in
# sync with //third_party/golang/esbuild/version.txt in google3.
_ESBUILD_VERSION = "0.28.2"

# (os, arch) -> (npm package suffix, npm integrity).
_ESBUILD_PLATFORMS = {
    ("linux", "x86_64"): ("linux-x64", "sha512-4xTZr1FUmSoQW4XIWmit3tzQrUTZM+N3P0XV8xROKYF50XfI7xeO90+1bZvNwxIufQ9hDQVRJH5YhgPVF8A/HQ=="),
    ("linux", "aarch64"): ("linux-arm64", "sha512-pW4AC0P3it8c7do9MVM4p51FzHzdM/TZrerurgRcHJ2WTa1VQ1CIq18xncfpBJw4ojkiZZrKW2yIBWBP92j6Ug=="),
    ("mac os x", "x86_64"): ("darwin-x64", "sha512-uq6suIWYP37qzGddBKPw5QEQPi6HiLGsO7UmkpfyaYNQ3D+rN6w6WfwH+nuqcGXWvawGwxOEroO4YGnFh95azw=="),
    ("mac os x", "aarch64"): ("darwin-arm64", "sha512-n4KqkOQrraxHJcgjM1RvwbigfQKIKJVpM7xp+KsxiyUSrRdIXnt73VhrPAx0fV44hgfmIVKjxMN9J1t5jySVkw=="),
}

def _esbuild_prebuilt_impl(rctx):
    os = rctx.os.name.lower()
    os = "mac os x" if os.startswith("mac") else os
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(rctx.os.arch, rctx.os.arch)
    if (os, arch) not in _ESBUILD_PLATFORMS:
        fail("esbuild: unsupported host platform %s/%s" % (os, arch))
    suffix, integrity = _ESBUILD_PLATFORMS[(os, arch)]

    rctx.download_and_extract(
        url = "https://registry.npmjs.org/@esbuild/{suffix}/-/{suffix}-{version}.tgz".format(
            suffix = suffix,
            version = _ESBUILD_VERSION,
        ),
        integrity = integrity,
        stripPrefix = "package",
    )
    rctx.file("BUILD.bazel", """\
filegroup(
    name = "esbuild",
    srcs = ["bin/esbuild"],
    visibility = ["//visibility:public"],
)
""")

esbuild_prebuilt = repository_rule(
    implementation = _esbuild_prebuilt_impl,
)
