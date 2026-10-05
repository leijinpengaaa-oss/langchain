param(
    [int]$Port = 8765,
    [switch]$Offline
)

$deskRoot = $PSScriptRoot
$deskPython = Join-Path $deskRoot '.venv-demo\Scripts\python.exe'
$deskRunner = Join-Path $deskRoot 'run_knowledge_desk.py'
if (-not (Test-Path -LiteralPath $deskPython)) {
    throw '请先按照 apps/knowledge_desk/README.md 创建虚拟环境并安装依赖。'
}
$deskArguments = @('-X', 'utf8', $deskRunner, '--port', $Port)
if ($Offline) {
    $deskArguments += '--offline'
}
Write-Host "知行助手：http://127.0.0.1:$Port（Ctrl+C 停止）"
& $deskPython @deskArguments
exit $LASTEXITCODE
