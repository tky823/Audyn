from audyn.utils.audio import list_audio_backends


def main() -> None:
    is_available = "ffmpeg" in list_audio_backends()
    print(is_available)


if __name__ == "__main__":
    main()
