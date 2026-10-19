import sys
from argparse import ArgumentParser, Namespace

import torch
from packaging import version


def main() -> None:
    args = parse_args()

    torch_version = version.parse(torch.__version__)

    if torch_version >= version.parse("2.9"):
        torchcodec_version = "0" + "." + str(torch_version.minor)
        is_linux = sys.platform == "linux"

        with open(args.path, mode="w") as f:
            if args.cpu_only and torch_version >= version.parse("2.11") and is_linux:
                f.write("--index-url https://download.pytorch.org/whl/cpu\n")

            f.write("torchcodec==" + torchcodec_version + "\n")
    else:
        with open(args.path, mode="w") as f:
            pass


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Overwrite pyptoject.toml to specify version of torch.")

    parser.add_argument("--path", type=str, help="Path to requirements.txt")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Use the CPU-only PyTorch wheel index for TorchCodec.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
