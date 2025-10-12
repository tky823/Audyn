import os
import shutil
import tempfile
import uuid
from argparse import ArgumentParser, Namespace

from packaging import version


def main() -> None:
    args = parse_args()

    overwrite_pyproject_toml(args.path, torch_version=args.torch_version)


def overwrite_pyproject_toml(path: str, torch_version: str) -> None:
    unspecified = '"torch",'
    specified = '"torch=={}",'.format(torch_version)

    is_torch_lt_2_3 = version.parse(torch_version) < version.parse("2.3")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, str(uuid.uuid4()))

        with open(path) as f_in, open(temp_path, mode="w") as f_out:
            for line in f_in:
                if unspecified in line:
                    line = line.replace(unspecified, specified)
                elif '"numpy",' in line and is_torch_lt_2_3:
                    line = line.replace('"numpy",', '"numpy<2.0",')

                f_out.write(line)

        shutil.copyfile(temp_path, path)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Overwrite pyptoject.toml to specify version of torch.")

    parser.add_argument("--path", type=str, help="Path to pyproject.toml")
    parser.add_argument(
        "--torch-version",
        type=str,
        help="Version of torch specified as torch==<version>.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
