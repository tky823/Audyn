import os
import subprocess
import sys
import tempfile

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CppExtension

IS_WINDOWS = sys.platform == "win32"


def is_openmp_supported() -> bool:
    """Check if OpenMP is available."""
    compiler = get_cxx_compiler()
    is_supported = None

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", dir=temp_dir, delete=False) as f:
            cpp_text = """
            #include <omp.h>

            int main() {
                return 0;
            }
            """
            f.write(cpp_text)

        obj_name = f.name.replace(".cpp", ".out")

        try:
            subprocess.check_output([compiler, f.name, "-o", obj_name, "-fopenmp"])
            is_supported = True
        except subprocess.CalledProcessError:
            is_supported = False

    if is_supported is None:
        raise RuntimeError("Unexpected error happened while checking if OpenMP is available.")

    return is_supported


def is_flag_accepted(flag: str) -> bool:
    """Check if flag is available."""
    compiler = get_cxx_compiler()
    is_accepted = None

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", dir=temp_dir, delete=False) as f:
            cpp_text = """
            int main() {
                return 0;
            }
            """
            f.write(cpp_text)

        obj_name = f.name.replace(".cpp", ".out")

        try:
            subprocess.check_output([compiler, f.name, "-o", obj_name, flag])
            is_accepted = True
        except subprocess.CalledProcessError:
            is_accepted = False

    if is_accepted is None:
        raise RuntimeError(f"Unexpected error happened while checking if {flag} is available.")

    return is_accepted


def get_cxx_compiler() -> str:
    try:
        from torch.utils.cpp_extension import get_cxx_compiler as _get_cxx_compiler

        compiler = _get_cxx_compiler()
    except ImportError:
        if IS_WINDOWS:
            compiler = os.environ.get("CXX", "cl")
        else:
            compiler = os.environ.get("CXX", "c++")

    return compiler


class BuildExtension(_BuildExtension):
    cpp_extensions = [
        {
            "name": "audyn._cpp_extensions.monotonic_align",
            "sources": [
                "cpp_extensions/monotonic_align/monotonic_align.cpp",
            ],
            # add extra_compile_args and extra_link_args
            "extra_compile_args": [],
            "extra_link_args": [],
        },
    ]

    if is_flag_accepted("-O3"):
        for cpp_extension in cpp_extensions:
            if cpp_extension["name"] == "audyn._cpp_extensions.monotonic_align":
                cpp_extension["extra_compile_args"].append("-O3")

    if is_flag_accepted("-march=native"):
        for cpp_extension in cpp_extensions:
            if cpp_extension["name"] == "audyn._cpp_extensions.monotonic_align":
                cpp_extension["extra_compile_args"].append("-march=native")

    if is_openmp_supported():
        for cpp_extension in cpp_extensions:
            if cpp_extension["name"] == "audyn._cpp_extensions.monotonic_align":
                cpp_extension["extra_compile_args"].append("-fopenmp")
                cpp_extension["extra_link_args"].append("-fopenmp")

    def run(self) -> None:
        if self.editable_mode:
            # create directories to save ".so" files in editable mode.
            for cpp_extension in self.cpp_extensions:
                *pkg_names, _ = cpp_extension["name"].split(".")
                os.makedirs("/".join(pkg_names), exist_ok=True)

        super().run()


# NOTE: Basic settings are written in pyproject.toml.
setup(
    ext_modules=[CppExtension(**extension) for extension in BuildExtension.cpp_extensions],
    cmdclass={"build_ext": BuildExtension},
)
