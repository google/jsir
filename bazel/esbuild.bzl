"""A minimal `esbuild_binary` for the open-source build.

Mirrors the subset of the google3 macro (//third_party/golang/esbuild:build_defs.bzl)
that JSIR uses, so that the same BUILD files work in both places; copybara
rewrites the load() statement. esbuild itself is a prebuilt, self-contained
binary fetched by //bazel:extensions.bzl, so no Node.js toolchain or npm
packages are needed.
"""

def esbuild_binary(
        name,
        entrypoints,
        srcs,
        out,
        platform = "node",
        target = "node18",
        minify = False,
        inject_import_meta = True,  # @unused: Only meaningful in google3.
        **kwargs):
    """Bundles a single entrypoint (and everything it imports) with esbuild.

    Args:
        name: Name of the target.
        entrypoints: A list with exactly one entrypoint file.
        srcs: All sources reachable from the entrypoint via imports.
        out: The output bundle.
        platform: https://esbuild.github.io/api/#platform.
        target: https://esbuild.github.io/api/#target.
        minify: Whether to minify the output.
        inject_import_meta: Ignored.
        **kwargs: Passed to the underlying genrule.
    """
    if len(entrypoints) != 1:
        fail("esbuild_binary requires exactly one entrypoint")
    entrypoint = entrypoints[0]

    native.genrule(
        name = name,
        srcs = depset(srcs + entrypoints).to_list(),
        outs = [out],
        tools = ["@esbuild//:esbuild"],
        cmd = " ".join([
            "$(execpath @esbuild//:esbuild)",
            "$(execpath %s)" % entrypoint,
            "--bundle",
            "--log-level=warning",
            "--platform=%s" % platform,
            "--target=%s" % target,
            "--minify" if minify else "",
            "--outfile=$@",
        ]),
        **kwargs
    )
