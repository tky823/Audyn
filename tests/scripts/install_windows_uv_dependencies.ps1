# uv
$UV_PATH = "${env:USERPROFILE}\.tools\bin"
$env:UV_INSTALL_DIR = $UV_PATH
irm https://astral.sh/uv/install.ps1 | iex
echo $UV_PATH >> ${env:GITHUB_PATH}
