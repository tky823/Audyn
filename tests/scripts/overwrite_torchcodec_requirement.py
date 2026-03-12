from argparse import ArgumentParser, Namespace

import torch
from packaging import version


def main() -> None:
    args = parse_args()

    torch_version = version.parse(torch.__version__)

    if torch_version >= version.parse("2.9"):
        torchcodec_version = "0" + "." + torch_version.minor

        with open(args.path, mode="w") as f:
            f.write("torchcodec==" + torchcodec_version)
    else:
        with open(args.path, mode="w") as f:
            pass


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Overwrite pyptoject.toml to specify version of torch.")

    parser.add_argument("--path", type=str, help="Path to requirements.txt")

    return parser.parse_args()


if __name__ == "__main__":
    main()
