# uio installer for Windows:  irm https://github.com/michael-berardi/ultra-inference-optimization/releases/latest/download/install.ps1 | iex
# uio uses POSIX file locks and process-group signals, so on Windows it runs inside WSL (Windows Subsystem for Linux).
# This script downloads install.sh and SHA256SUMS from the same release, verifies install.sh's SHA-256, then runs the
# Linux installer inside WSL. Environment: UIO_RELEASE_BASE and UIO_VERSION override the release source.
# The body is a function so `irm | iex` does not leak variables into the caller's session, and it throws instead of
# calling `exit`, which would close the user's own PowerShell window before the instructions can be read.
function Install-Uio {
    $ErrorActionPreference = 'Stop'
    $repo = 'github.com/michael-berardi/ultra-inference-optimization'
    $base = if ($env:UIO_RELEASE_BASE) {
        $env:UIO_RELEASE_BASE.TrimEnd('/')
    } elseif ($env:UIO_VERSION) {
        "https://$repo/releases/download/v$($env:UIO_VERSION.TrimStart('v'))"
    } else {
        "https://$repo/releases/latest/download"
    }

    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'uio on Windows runs inside WSL, which is not installed. Run `wsl --install -d Ubuntu` in an administrator PowerShell, restart, then open Ubuntu once to finish setup.'
    }
    & wsl.exe --status *> $null
    & wsl.exe -e true *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'WSL is installed but has no Linux distribution that starts. Run `wsl --install -d Ubuntu`, open Ubuntu once to finish its setup, then run this installer again.'
    }

    $saved = @{ WSLENV = $env:WSLENV; Base = $env:UIO_RELEASE_BASE; EAP = $ErrorActionPreference }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("uio-install-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Invoke-WebRequest "$base/install.sh" -OutFile (Join-Path $tmp 'install.sh') -UseBasicParsing
        Invoke-WebRequest "$base/SHA256SUMS" -OutFile (Join-Path $tmp 'SHA256SUMS') -UseBasicParsing
        $want = (Get-Content (Join-Path $tmp 'SHA256SUMS') | Where-Object { $_ -match '\sinstall\.sh$' } | Select-Object -First 1) -split '\s+' | Select-Object -First 1
        $got = (Get-FileHash (Join-Path $tmp 'install.sh') -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $want -or $got -ne $want.ToLowerInvariant()) { throw 'install.sh does not match its checksum; nothing was installed.' }

        $linuxPath = (& wsl.exe -e wslpath -a ((Join-Path $tmp 'install.sh') -replace '\\', '/')).Trim()
        $env:WSLENV = if ($saved.WSLENV) { "UIO_RELEASE_BASE/u:$($saved.WSLENV)" } else { 'UIO_RELEASE_BASE/u' }
        $env:UIO_RELEASE_BASE = $base
        & wsl.exe -e sh $linuxPath
        if ($LASTEXITCODE -ne 0) { throw "The Linux installer failed (exit $LASTEXITCODE)." }
        Write-Host 'uio is installed in WSL. Open your Linux terminal (or run `wsl`) and try: uio probe --no-measure'
    } finally {
        $env:WSLENV = $saved.WSLENV
        $env:UIO_RELEASE_BASE = $saved.Base
        $ErrorActionPreference = $saved.EAP
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}
Install-Uio
