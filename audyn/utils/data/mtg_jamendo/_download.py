import csv
import os
from typing import Any, Dict, List, Optional

from ... import audyn_cache_dir
from ..download import download_file


def download_all_tags() -> List[str]:
    """Download tags of 'all' set."""
    tags = [
        "genre---60s",
        "genre---70s",
        "genre---80s",
        "genre---90s",
        "genre---acidjazz",
        "genre---alternative",
        "genre---alternativerock",
        "genre---ambient",
        "genre---atmospheric",
        "genre---blues",
        "genre---bluesrock",
        "genre---bossanova",
        "genre---breakbeat",
        "genre---celtic",
        "genre---chanson",
        "genre---chillout",
        "genre---choir",
        "genre---classical",
        "genre---classicrock",
        "genre---club",
        "genre---contemporary",
        "genre---country",
        "genre---dance",
        "genre---darkambient",
        "genre---darkwave",
        "genre---deephouse",
        "genre---disco",
        "genre---downtempo",
        "genre---drumnbass",
        "genre---dub",
        "genre---dubstep",
        "genre---easylistening",
        "genre---edm",
        "genre---electronic",
        "genre---electronica",
        "genre---electropop",
        "genre---ethno",
        "genre---eurodance",
        "genre---experimental",
        "genre---folk",
        "genre---funk",
        "genre---fusion",
        "genre---groove",
        "genre---grunge",
        "genre---hard",
        "genre---hardrock",
        "genre---hiphop",
        "genre---house",
        "genre---idm",
        "genre---improvisation",
        "genre---indie",
        "genre---industrial",
        "genre---instrumentalpop",
        "genre---instrumentalrock",
        "genre---jazz",
        "genre---jazzfusion",
        "genre---latin",
        "genre---lounge",
        "genre---medieval",
        "genre---metal",
        "genre---minimal",
        "genre---newage",
        "genre---newwave",
        "genre---orchestral",
        "genre---pop",
        "genre---popfolk",
        "genre---poprock",
        "genre---postrock",
        "genre---progressive",
        "genre---psychedelic",
        "genre---punkrock",
        "genre---rap",
        "genre---reggae",
        "genre---rnb",
        "genre---rock",
        "genre---rocknroll",
        "genre---singersongwriter",
        "genre---soul",
        "genre---soundtrack",
        "genre---swing",
        "genre---symphonic",
        "genre---synthpop",
        "genre---techno",
        "genre---trance",
        "genre---triphop",
        "genre---world",
        "genre---worldfusion",
        "instrument---accordion",
        "instrument---acousticbassguitar",
        "instrument---acousticguitar",
        "instrument---bass",
        "instrument---beat",
        "instrument---bell",
        "instrument---bongo",
        "instrument---brass",
        "instrument---cello",
        "instrument---clarinet",
        "instrument---classicalguitar",
        "instrument---computer",
        "instrument---doublebass",
        "instrument---drummachine",
        "instrument---drums",
        "instrument---electricguitar",
        "instrument---electricpiano",
        "instrument---flute",
        "instrument---guitar",
        "instrument---harmonica",
        "instrument---harp",
        "instrument---horn",
        "instrument---keyboard",
        "instrument---oboe",
        "instrument---orchestra",
        "instrument---organ",
        "instrument---pad",
        "instrument---percussion",
        "instrument---piano",
        "instrument---pipeorgan",
        "instrument---rhodes",
        "instrument---sampler",
        "instrument---saxophone",
        "instrument---strings",
        "instrument---synthesizer",
        "instrument---trombone",
        "instrument---trumpet",
        "instrument---viola",
        "instrument---violin",
        "instrument---voice",
        "mood/theme---action",
        "mood/theme---adventure",
        "mood/theme---advertising",
        "mood/theme---background",
        "mood/theme---ballad",
        "mood/theme---calm",
        "mood/theme---children",
        "mood/theme---christmas",
        "mood/theme---commercial",
        "mood/theme---cool",
        "mood/theme---corporate",
        "mood/theme---dark",
        "mood/theme---deep",
        "mood/theme---documentary",
        "mood/theme---drama",
        "mood/theme---dramatic",
        "mood/theme---dream",
        "mood/theme---emotional",
        "mood/theme---energetic",
        "mood/theme---epic",
        "mood/theme---fast",
        "mood/theme---film",
        "mood/theme---fun",
        "mood/theme---funny",
        "mood/theme---game",
        "mood/theme---groovy",
        "mood/theme---happy",
        "mood/theme---heavy",
        "mood/theme---holiday",
        "mood/theme---hopeful",
        "mood/theme---inspiring",
        "mood/theme---love",
        "mood/theme---meditative",
        "mood/theme---melancholic",
        "mood/theme---melodic",
        "mood/theme---motivational",
        "mood/theme---movie",
        "mood/theme---nature",
        "mood/theme---party",
        "mood/theme---positive",
        "mood/theme---powerful",
        "mood/theme---relaxing",
        "mood/theme---retro",
        "mood/theme---romantic",
        "mood/theme---sad",
        "mood/theme---sexy",
        "mood/theme---slow",
        "mood/theme---soft",
        "mood/theme---soundscape",
        "mood/theme---space",
        "mood/theme---sport",
        "mood/theme---summer",
        "mood/theme---trailer",
        "mood/theme---travel",
        "mood/theme---upbeat",
        "mood/theme---uplifting",
    ]

    return tags


def download_top50_tags() -> List[str]:
    """Download tags of 'top50' set."""
    tags = [
        "genre---alternative",
        "genre---ambient",
        "genre---atmospheric",
        "genre---chillout",
        "genre---classical",
        "genre---dance",
        "genre---downtempo",
        "genre---easylistening",
        "genre---electronic",
        "genre---experimental",
        "genre---folk",
        "genre---funk",
        "genre---hiphop",
        "genre---house",
        "genre---indie",
        "genre---instrumentalpop",
        "genre---jazz",
        "genre---lounge",
        "genre---metal",
        "genre---newage",
        "genre---orchestral",
        "genre---pop",
        "genre---popfolk",
        "genre---poprock",
        "genre---reggae",
        "genre---rock",
        "genre---soundtrack",
        "genre---techno",
        "genre---trance",
        "genre---triphop",
        "genre---world",
        "instrument---acousticguitar",
        "instrument---bass",
        "instrument---computer",
        "instrument---drummachine",
        "instrument---drums",
        "instrument---electricguitar",
        "instrument---electricpiano",
        "instrument---guitar",
        "instrument---keyboard",
        "instrument---piano",
        "instrument---strings",
        "instrument---synthesizer",
        "instrument---violin",
        "instrument---voice",
        "mood/theme---emotional",
        "mood/theme---energetic",
        "mood/theme---film",
        "mood/theme---happy",
        "mood/theme---relaxing",
    ]

    return tags


def download_genre_tags() -> List[str]:
    """Download tags for 'genre' category."""
    tags = [
        "genre---60s",
        "genre---70s",
        "genre---80s",
        "genre---90s",
        "genre---acidjazz",
        "genre---african",
        "genre---alternative",
        "genre---alternativerock",
        "genre---ambient",
        "genre---atmospheric",
        "genre---blues",
        "genre---bluesrock",
        "genre---bossanova",
        "genre---breakbeat",
        "genre---celtic",
        "genre---chanson",
        "genre---chillout",
        "genre---choir",
        "genre---classical",
        "genre---classicrock",
        "genre---club",
        "genre---contemporary",
        "genre---country",
        "genre---dance",
        "genre---darkambient",
        "genre---darkwave",
        "genre---deephouse",
        "genre---disco",
        "genre---downtempo",
        "genre---drumnbass",
        "genre---dub",
        "genre---dubstep",
        "genre---easylistening",
        "genre---edm",
        "genre---electronic",
        "genre---electronica",
        "genre---electropop",
        "genre---ethnicrock",
        "genre---ethno",
        "genre---eurodance",
        "genre---experimental",
        "genre---folk",
        "genre---funk",
        "genre---fusion",
        "genre---gothic",
        "genre---groove",
        "genre---grunge",
        "genre---hard",
        "genre---hardrock",
        "genre---heavymetal",
        "genre---hiphop",
        "genre---house",
        "genre---idm",
        "genre---improvisation",
        "genre---indie",
        "genre---industrial",
        "genre---instrumentalpop",
        "genre---instrumentalrock",
        "genre---jazz",
        "genre---jazzfunk",
        "genre---jazzfusion",
        "genre---latin",
        "genre---lounge",
        "genre---medieval",
        "genre---metal",
        "genre---minimal",
        "genre---newage",
        "genre---newwave",
        "genre---orchestral",
        "genre---oriental",
        "genre---pop",
        "genre---popfolk",
        "genre---poprock",
        "genre---postrock",
        "genre---progressive",
        "genre---psychedelic",
        "genre---punkrock",
        "genre---rap",
        "genre---reggae",
        "genre---rnb",
        "genre---rock",
        "genre---rocknroll",
        "genre---singersongwriter",
        "genre---ska",
        "genre---soul",
        "genre---soundtrack",
        "genre---swing",
        "genre---symphonic",
        "genre---synthpop",
        "genre---techno",
        "genre---trance",
        "genre---tribal",
        "genre---triphop",
        "genre---world",
        "genre---worldfusion",
    ]

    return tags


def download_instrument_tags() -> List[str]:
    """Download tags for 'instrument' category."""
    tags = [
        "instrument---accordion",
        "instrument---acousticbassguitar",
        "instrument---acousticguitar",
        "instrument---bass",
        "instrument---beat",
        "instrument---bell",
        "instrument---bongo",
        "instrument---brass",
        "instrument---cello",
        "instrument---clarinet",
        "instrument---classicalguitar",
        "instrument---computer",
        "instrument---doublebass",
        "instrument---drummachine",
        "instrument---drums",
        "instrument---electricguitar",
        "instrument---electricpiano",
        "instrument---flute",
        "instrument---guitar",
        "instrument---harmonica",
        "instrument---harp",
        "instrument---horn",
        "instrument---keyboard",
        "instrument---oboe",
        "instrument---orchestra",
        "instrument---organ",
        "instrument---pad",
        "instrument---percussion",
        "instrument---piano",
        "instrument---pipeorgan",
        "instrument---rhodes",
        "instrument---sampler",
        "instrument---saxophone",
        "instrument---strings",
        "instrument---synthesizer",
        "instrument---trombone",
        "instrument---trumpet",
        "instrument---ukulele",
        "instrument---viola",
        "instrument---violin",
        "instrument---voice",
    ]

    return tags


def download_moodtheme_tags() -> List[str]:
    """Download tags for 'mood/theme' category."""
    tags = [
        "mood/theme---action",
        "mood/theme---adventure",
        "mood/theme---advertising",
        "mood/theme---ambiental",
        "mood/theme---background",
        "mood/theme---ballad",
        "mood/theme---calm",
        "mood/theme---children",
        "mood/theme---christmas",
        "mood/theme---commercial",
        "mood/theme---cool",
        "mood/theme---corporate",
        "mood/theme---dark",
        "mood/theme---deep",
        "mood/theme---documentary",
        "mood/theme---drama",
        "mood/theme---dramatic",
        "mood/theme---dream",
        "mood/theme---emotional",
        "mood/theme---energetic",
        "mood/theme---epic",
        "mood/theme---fast",
        "mood/theme---film",
        "mood/theme---fun",
        "mood/theme---funny",
        "mood/theme---game",
        "mood/theme---groovy",
        "mood/theme---happy",
        "mood/theme---heavy",
        "mood/theme---holiday",
        "mood/theme---hopeful",
        "mood/theme---horror",
        "mood/theme---inspiring",
        "mood/theme---love",
        "mood/theme---meditative",
        "mood/theme---melancholic",
        "mood/theme---mellow",
        "mood/theme---melodic",
        "mood/theme---motivational",
        "mood/theme---movie",
        "mood/theme---nature",
        "mood/theme---party",
        "mood/theme---positive",
        "mood/theme---powerful",
        "mood/theme---relaxing",
        "mood/theme---retro",
        "mood/theme---romantic",
        "mood/theme---sad",
        "mood/theme---sexy",
        "mood/theme---slow",
        "mood/theme---soft",
        "mood/theme---soundscape",
        "mood/theme---space",
        "mood/theme---sport",
        "mood/theme---summer",
        "mood/theme---trailer",
        "mood/theme---travel",
        "mood/theme---upbeat",
        "mood/theme---uplifting",
    ]

    return tags


def download_all_metadata(
    root: Optional[str] = None,
    split: int = 0,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, Any]]:
    base_url = f"https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-{split}"  # noqa: E501
    base_url += "/autotagging-{subset}.tsv"
    metadata = []

    for subset in ["train", "validation", "test"]:
        url = base_url.format(subset=subset)
        filename = "{subset}.tsv".format(subset=subset)

        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo", "all", f"split-{split}")

        path = os.path.join(root, filename)

        download_file(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )
        _metadata = _load_metadata(path, subset=subset)
        metadata.extend(_metadata)

    return metadata


def download_top50_metadata(
    root: Optional[str] = None,
    split: int = 0,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, Any]]:
    base_url = f"https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-{split}"  # noqa: E501
    base_url += "/autotagging_top50tags-{subset}.tsv"  # noqa: E501
    metadata = []

    for subset in ["train", "validation", "test"]:
        url = base_url.format(subset=subset)
        filename = "{subset}.tsv".format(subset=subset)

        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo", "top50", f"split-{split}")

        path = os.path.join(root, filename)

        download_file(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )
        _metadata = _load_metadata(path, subset=subset)
        metadata.extend(_metadata)

    return metadata


def download_genre_metadata(
    root: Optional[str] = None,
    split: int = 0,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, Any]]:
    base_url = f"https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-{split}"  # noqa: E501
    base_url += "/autotagging_genre-{subset}.tsv"  # noqa: E501
    metadata = []

    for subset in ["train", "validation", "test"]:
        url = base_url.format(subset=subset)
        filename = "{subset}.tsv".format(subset=subset)

        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo", "genre", f"split-{split}")

        path = os.path.join(root, filename)

        download_file(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )
        _metadata = _load_metadata(path, subset=subset)
        metadata.extend(_metadata)

    return metadata


def download_instrument_metadata(
    root: Optional[str] = None,
    split: int = 0,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, Any]]:
    base_url = f"https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-{split}"  # noqa: E501
    base_url += "/autotagging_instrument-{subset}.tsv"  # noqa: E501
    metadata = []

    for subset in ["train", "validation", "test"]:
        url = base_url.format(subset=subset)
        filename = "{subset}.tsv".format(subset=subset)

        if root is None:
            root = os.path.join(
                audyn_cache_dir, "data", "mtg-jamendo", "instrument", f"split-{split}"
            )

        path = os.path.join(root, filename)

        download_file(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )
        _metadata = _load_metadata(path, subset=subset)
        metadata.extend(_metadata)

    return metadata


def download_moodtheme_metadata(
    root: Optional[str] = None,
    split: int = 0,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, Any]]:
    base_url = f"https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-{split}"  # noqa: E501
    base_url += "/autotagging_moodtheme-{subset}.tsv"  # noqa: E501
    metadata = []

    for subset in ["train", "validation", "test"]:
        url = base_url.format(subset=subset)
        filename = "{subset}.tsv".format(subset=subset)

        if root is None:
            root = os.path.join(
                audyn_cache_dir, "data", "mtg-jamendo", "moodtheme", f"split-{split}"
            )

        path = os.path.join(root, filename)

        download_file(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )
        _metadata = _load_metadata(path, subset=subset)
        metadata.extend(_metadata)

    return metadata


def _load_metadata(path: str, subset: Optional[str] = None) -> List[Dict[str, Any]]:
    metadata = []

    with open(path) as f:
        for idx, (track, artist, album, _path, duration, *tags) in enumerate(
            csv.reader(f, delimiter="\t")
        ):
            if idx < 1:
                continue

            data = {
                "track": track,
                "artist": artist,
                "album": album,
                "path": _path,
                "duration": float(duration),
                "tags": list(tags),
            }

            if subset is not None:
                data["subset"] = subset

            metadata.append(data)

    return metadata
