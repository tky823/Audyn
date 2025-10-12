# uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# FFmpeg
choco install ffmpeg --version=6.1.0
echo "C:\ProgramData\chocolatey\bin" >> ${env:GITHUB_PATH}
