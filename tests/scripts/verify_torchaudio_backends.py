from audyn.utils.audio import list_audio_backends


def main() -> None:
    assert len(list_audio_backends()) > 0


if __name__ == "__main__":
    main()
