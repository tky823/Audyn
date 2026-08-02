import os
import subprocess
import sys
import tempfile

import torch
from packaging import version
from setuptools import setup
from setuptools.extension import Extension
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CppExtension

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform.startswith("darwin")
IS_LINUX = sys.platform.startswith("linux")

IS_TORCH_GE_2_4 = version.parse(torch.__version__) >= version.parse("2.4")

SUBPROCESS_DECODE_ARGS = ("oem",) if IS_WINDOWS else ()


def get_openmp_flags(compiler: str) -> tuple[bool, list[str], list[str]]:
    """
    Check if OpenMP is available.
    Returns: (is_supported, compile_flags, link_flags)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        cpp_file = os.path.join(temp_dir, "test.cpp")
        with open(cpp_file, "w") as f:
            f.write("#include <omp.h>\nint main() { return 0; }\n")

        if compiler == "cl":
            cflags = ["/openmp"]
            ldflags = []
        elif IS_MACOS:
            # Apple Clang requires these specific flags for OpenMP
            cflags = ["-Xpreprocessor", "-fopenmp"]
            ldflags = ["-lomp"]
        else:
            cflags = ["-fopenmp"]
            ldflags = ["-fopenmp"]

        cmd = [compiler, cpp_file] + cflags + ldflags

        try:
            # Suppress output for clean installation logs
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, cflags, ldflags
        except subprocess.CalledProcessError:
            return False, [], []


def is_flag_accepted(compiler: str, flag: str) -> bool:
    """Check if a specific compiler flag is available."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cpp_file = os.path.join(temp_dir, "test.cpp")
        with open(cpp_file, "w") as f:
            f.write("int main() { return 0; }\n")

        try:
            # Simply attempt to compile the empty file with the given flag
            subprocess.check_call(
                [compiler, cpp_file, flag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False


def get_cxx_compiler() -> str:
    compiler = None

    try:
        from torch.utils.cpp_extension import get_cxx_compiler as _get_cxx_compiler

        compiler = _get_cxx_compiler()
    except ImportError:
        if IS_WINDOWS:
            compiler = os.environ.get("CXX", "cl")
        else:
            compiler = os.environ.get("CXX", "c++")

    if compiler is None:
        raise RuntimeError("Unexpected error happened while checking cxx compiler.")

    return compiler


class BuildExtension(_BuildExtension):
    cpp_extensions = []

    if IS_TORCH_GE_2_4:
        cpp_extensions.append(
            {
                "name": "audyn._C.monotonic_align",
                "sources": [
                    "csrc/monotonic_align_torch_2_4.cpp",
                ],
            },
        )
    else:
        cpp_extensions.append(
            {
                "name": "audyn._C.monotonic_align",
                "sources": [
                    "csrc/monotonic_align.cpp",
                ],
            },
        )

    cpp_extensions.append(
        {
            "name": "audyn._C.bipartite_match",
            "sources": [
                "csrc/bipartite_match.cpp",
            ],
        },
    )

    def build_extension(self, ext: Extension) -> None:
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()

        # Fix for older PyTorch versions on macOS (is_arithmetic error)
        if IS_MACOS and is_flag_accepted(compiler, "-Wno-invalid-specialization"):
            ext.extra_compile_args.append("-Wno-invalid-specialization")

        if ext.name == "audyn._C.monotonic_align" and not IS_WINDOWS:
            # TODO: support Windows
            which = subprocess.check_output(["which", compiler], stderr=subprocess.STDOUT)
            compiler = os.path.realpath(which.decode(*SUBPROCESS_DECODE_ARGS).strip())

            # optimization
            if is_flag_accepted(compiler, "-O3"):
                ext.extra_compile_args.append("-O3")

            # environment-dependent optimization
            if is_flag_accepted(compiler, "-march=native"):
                ext.extra_compile_args.append("-march=native")

            # availability of OpenMP
            is_omp_supported, omp_cflags, omp_ldflags = get_openmp_flags(compiler)

            if is_omp_supported:
                ext.extra_compile_args.extend(omp_cflags)
                ext.extra_link_args.extend(omp_ldflags)

        return super().build_extension(ext)


# NOTE: Basic settings are written in pyproject.toml.
setup(
    ext_modules=[CppExtension(**extension) for extension in BuildExtension.cpp_extensions],
    cmdclass={"build_ext": BuildExtension},
)
