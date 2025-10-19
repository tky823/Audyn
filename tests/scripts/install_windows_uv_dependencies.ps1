# uv
$env:UV_NO_MODIFY_PATH = "1"
$env:UV_INSTALL_DIR = "${env:USERPROFILE}\.uv\bin"
irm https://astral.sh/uv/install.ps1 | iex
echo "${env:UV_INSTALL_DIR}" >> ${env:GITHUB_PATH}
