import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension as _BuildExtension
from torch.utils.cpp_extension import CppExtension


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


# NOTE: Basic settings are written in pyproject.toml.
setup(
    ext_modules=[CppExtension(**extension) for extension in BuildExtension.cpp_extensions],
    cmdclass={"build_ext": BuildExtension},
)
