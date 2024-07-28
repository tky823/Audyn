from omegaconf import DictConfig

import audyn


@audyn.main(config_name="config_download-mtg-jamendo")
def main(config: DictConfig) -> None:
    """Download MTG-Jamendo audio files.

    .. code-block:: shell

        quality="raw"  # or "low"
        server_type="mirror"  # or "origin"

        audyn-download-mtg-jamando \
        quality="${quality}" \
        server_type="${server_type}"

    """
    quality = config.quality
    server_type = config.server_type

    assert quality in ["raw", "low"], f"{quality} is not supported as quality. Use 'raw' or 'low'."
    assert server_type in [
        "origin",
        "mirror",
    ], f"{server_type} is not supported as quality. Use 'origin' or 'mirror'."


if __name__ == "__main__":
    main()
