import torchaudio


def main() -> None:
    is_available = "ffmpeg" in torchaudio.list_audio_backends()
    print(is_available)


if __name__ == "__main__":
    main()
