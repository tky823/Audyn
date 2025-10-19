# uv
$env:UV_NO_MODIFY_PATH = "1"
irm https://astral.sh/uv/install.ps1 | iex
echo "${env:USERPROFILE}\.uv\bin" >> ${env:GITHUB_PATH}
