#!/bin/bash

# FFmpeg
brew install ffmpeg@6
homebrew_prefix="$(brew --prefix)"
echo "DYLD_FALLBACK_LIBRARY_PATH=${homebrew_prefix}/opt/ffmpeg@6/lib" >> "${GITHUB_ENV}"
echo "${homebrew_prefix}/opt/ffmpeg@6/bin" >> "${GITHUB_PATH}"

# OpenMP
brew install libomp
