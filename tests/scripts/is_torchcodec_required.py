import torch
from packaging import version


def main() -> None:
    if version.parse(torch.__version__) >= version.parse("2.9"):
        print("true")
    else:
        print("false")


if __name__ == "__main__":
    main()
