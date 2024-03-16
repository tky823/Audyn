import os
import subprocess
import sys
import tempfile

from setuptools import setup
from setuptools.extension import Extension
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CppExtension

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform.startswith("darwin")
IS_LINUX = sys.platform.startswith("linux")


SUBPROCESS_DECODE_ARGS = ("oem",) if IS_WINDOWS else ()


def is_openmp_supported(compiler: str) -> bool:
    """Check if OpenMP is available."""
    is_supported = None

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", dir=temp_dir) as f:
            cpp_text = """
            #include <omp.h>

            int main() {
                return 0;
            }
            """
            f.write(cpp_text)

            try:
                if compiler == "cl":
                    flag = "/openmp"
                else:
                    flag = "-fopenmp"

                subprocess.check_output([compiler, f.name, flag])
                is_supported = True
            except subprocess.CalledProcessError:
                is_supported = False

    if is_supported is None:
        raise RuntimeError("Unexpected error happened while checking if OpenMP is available.")

    return is_supported


def is_flag_accepted(compiler: str, flag: str) -> bool:
    """Check if flag is available."""
    is_accepted = None

    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", dir=temp_dir) as f:
            cpp_text = """
            int main() {
                return 0;
            }
            """
            f.write(cpp_text)

            try:
                if compiler == "cl":
                    subprocess.check_output([compiler])
                else:
                    subprocess.check_output([compiler, f.name, flag])

                is_accepted = True
            except subprocess.CalledProcessError:
                is_accepted = False

    if is_accepted is None:
        raise RuntimeError(f"Unexpected error happened while checking if {flag} is available.")

    return is_accepted


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
    cpp_extensions = [
        {
            "name": "audyn._cpp_extensions.monotonic_align",
            "sources": [
                "cpp_extensions/monotonic_align/monotonic_align.cpp",
            ],
        },
    ]

    def run(self) -> None:
        if self.editable_mode:
            # create directories to save ".so" files in editable mode.
            for cpp_extension in self.cpp_extensions:
                *pkg_names, _ = cpp_extension["name"].split(".")
                os.makedirs("/".join(pkg_names), exist_ok=True)

        super().run()

    def build_extension(self, ext: Extension) -> None:
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()

        if ext.name == "audyn._cpp_extensions.monotonic_align" and not IS_WINDOWS:
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
            if is_openmp_supported(compiler):
                ext.extra_compile_args.append("-fopenmp")
                ext.extra_link_args.append("-fopenmp")

        return super().build_extension(ext)


# NOTE: Basic settings are written in pyproject.toml.
setup(
    ext_modules=[CppExtension(**extension) for extension in BuildExtension.cpp_extensions],
    cmdclass={"build_ext": BuildExtension},
)
