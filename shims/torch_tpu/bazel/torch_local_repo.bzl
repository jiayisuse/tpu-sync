"""Expose the locally installed torch wheel and its C++ headers to Bazel."""

def _torch_local_repo_impl(rctx):
    torch_path = rctx.getenv("TORCH_SOURCE", "")
    if not torch_path:
        fail("TORCH_SOURCE must name the site-packages directory containing torch/")
    rctx.execute(["mkdir", "-p", "site-packages/torch"])
    rctx.execute(["cp", "-as", torch_path + "/torch/.", "site-packages/torch/"])
    if not rctx.path("site-packages/torch/include/torch/csrc/api/include/torch/torch.h").exists:
        fail("no torch headers under " + torch_path + "/torch/include")
    rctx.execute(["find", "site-packages", "(", "-name", "BUILD", "-o", "-name", "BUILD.bazel", ")", "-delete"])
    rctx.file("BUILD.bazel", """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_headers",
    hdrs = glob(
        [
            "site-packages/torch/include/**/*.h",
            "site-packages/torch/include/**/*.hpp",
            "site-packages/torch/include/**/*.cuh",
            "site-packages/torch/include/**/*.inc",
            "site-packages/torch/include/**/*.inl",
            "site-packages/torch/include/**/*.tcc",
        ],
        allow_empty = True,
    ),
    includes = [
        "site-packages/torch/include",
        "site-packages/torch/include/torch/csrc/api/include",
        "site-packages/torch/include/kineto",
    ],
    deps = ["@com_google_protobuf//:protobuf"],
)

py_library(name = "torch")
""")

torch_local_repo = repository_rule(
    implementation = _torch_local_repo_impl,
    environ = ["TORCH_SOURCE"],
    local = True,
)
