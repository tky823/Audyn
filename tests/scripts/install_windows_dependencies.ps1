# uv
$env:UV_NO_MODIFY_PATH = "1"
$env:UV_INSTALL_DIR = "${env:USERPROFILE}\.uv\bin"
irm https://astral.sh/uv/install.ps1 | iex
echo "${env:UV_INSTALL_DIR}" >> ${env:GITHUB_PATH}

# FFmpeg
choco install ffmpeg-shared --version=6.1.0
$ffmpegBin = Join-Path $env:ProgramFiles "FFmpeg\bin"

if (-not (Test-Path (Join-Path $ffmpegBin "ffmpeg.exe"))) {
  throw "Could not find ffmpeg.exe in $ffmpegBin."
}

$env:PATH = "$ffmpegBin;$env:PATH"
echo "$ffmpegBin" >> ${env:GITHUB_PATH}

ffmpeg -version
