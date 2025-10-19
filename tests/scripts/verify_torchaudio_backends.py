import torchaudio
from packaging import version


def main() -> None:
    if version.parse(torchaudio.__version__) < version.parse("2.9"):
        assert len(torchaudio.list_audio_backends()) > 0


if __name__ == "__main__":
    main()
