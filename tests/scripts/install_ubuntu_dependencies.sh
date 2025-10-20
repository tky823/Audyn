#!/bin/bash

sudo apt update

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "${HOME}/.uv/bin" >> ${GITHUB_PATH}

# FFmpeg
sudo apt install ffmpeg
