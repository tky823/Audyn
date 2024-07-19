# from .mp4 to .wav
# ported from https://github.com/sigsep/sigsep-mus-io/blob/023615154797df0c7b40005f773fc9d977c19915/scripts/decode.sh.

musdb18_root=""

. ../../_common/parse_options.sh || exit 1;

if [ -z "${musdb18_root}" ]; then
    echo "musdb18_root is not set."
    exit 1;
fi

subset_names=(train test)

(
    cd ${musdb18_root}

    for subset_name in "${subset_names[@]}" ; do
        cd ${subset_name}

        for stem in *.stem.mp4 ; do
            name="$(echo $stem | awk -F".stem.mp4" '{$0=$1}1')";

            if [ -e "${name}/mixture.wav" ] && [ -e "${name}/drums.wav" ] && [ -e "${name}/bass.wav" ] && [ -e "${name}/other.wav" ] && [ -e "${name}/vocals.wav" ]; then
                continue
            fi

            mkdir -p "${name}"
            cd "${name}"

            ffmpeg -loglevel panic -i "../${stem}" -map 0:0 -vn mixture.wav
            ffmpeg -loglevel panic -i "../${stem}" -map 0:1 -vn drums.wav
            ffmpeg -loglevel panic -i "../${stem}" -map 0:2 -vn bass.wav
            ffmpeg -loglevel panic -i "../${stem}" -map 0:3 -vn other.wav
            ffmpeg -loglevel panic -i "../${stem}" -map 0:4 -vn vocals.wav

            cd ../
        done

        cd ../
    done
)
