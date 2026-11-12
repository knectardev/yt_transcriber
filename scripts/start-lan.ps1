# Start the transcription app on this PC so other devices on the LAN can open it.
# Stop it with Ctrl+C. This script does not change the firewall or the network profile.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Missing $Python. Create the virtual environment before starting the LAN server."
}

$LogDir = Join-Path $Root ".local"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir "lan-ui.log"

$already = netstat -ano -p tcp | Select-String -Pattern "LISTENING" | Select-String -Pattern ":8765\s"
if ($already) {
    $lanIp = $null
    $udp = $null
    try {
        $udp = New-Object System.Net.Sockets.UdpClient
        $udp.Connect("8.8.8.8", 1)
        $lanIp = $udp.Client.LocalEndPoint.Address.ToString()
    } catch {
        $lanIp = $null
    } finally {
        if ($udp) { $udp.Close() }
    }
    Write-Host "The transcription server is already running."
    Write-Host "On this PC: http://127.0.0.1:8765"
    if ($lanIp -and $lanIp -notlike "169.254.*") {
        Write-Host "On your phone: http://${lanIp}:8765"
    }
    Write-Host "Closing this window does not stop it."
    exit 0
}

$env:BAROQUE_UI_ADDRESS = "0.0.0.0"
if (-not $env:BAROQUE_PLAYBACK_PORT) {
    $env:BAROQUE_PLAYBACK_PORT = "8766"
}
$env:PYTHONUNBUFFERED = "1"

Write-Host "Logging to $Log"
Write-Host "Stop the server with Ctrl+C in this window."
& $Python -m baroque_transcribe.cli ui --address 0.0.0.0 --port 8765 2>&1 | ForEach-Object {
    $line = $_.ToString()
    Write-Host $line
    Add-Content -Path $Log -Value $line
}
exit $LASTEXITCODE
