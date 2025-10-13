# uv
irm https://astral.sh/uv/install.ps1 | iex
echo "${env:USERPROFILE}\.uv\bin" >> ${env:GITHUB_PATH}

# FFmpeg
choco install ffmpeg --version=6.1.0
echo "C:\ProgramData\chocolatey\bin" >> ${env:GITHUB_PATH}
ffmpeg -version
ls "C:\ProgramData\chocolatey\bin"
