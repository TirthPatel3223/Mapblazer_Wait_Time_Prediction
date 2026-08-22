<#
.SYNOPSIS
    Load .env into the current PowerShell session and refresh PATH.

.DESCRIPTION
    The runbook was written in bash. On PowerShell `export FOO=bar` is a parse error and
    `$env:FOO = "bar"` is the equivalent, which is a papercut that costs a deploy attempt
    and, worse, invites pasting a live token onto the command line where it lands in
    PSReadLine history in cleartext. This reads the same gitignored .env the Python jobs
    read, so the token is never typed and there is exactly one place credentials live.

    Also re-reads PATH from the registry. A winget install modifies the persisted PATH but
    cannot touch an already-running shell, so a freshly installed CLI is "not recognized"
    until you restart. Re-reading it here saves the restart.

    Mirrors src/themepark/envfile.py: an already-set variable wins unless -Force, so this
    cannot silently override something you deliberately exported for one command.

.EXAMPLE
    . .\scripts\Load-Env.ps1
    databricks bundle validate --target prod

.NOTES
    Dot-source it. Running it as `.\scripts\Load-Env.ps1` sets the variables in a child
    scope that exits immediately, which looks like it did nothing.
#>
[CmdletBinding()]
param(
    [string] $Path,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'

if (-not $Path) {
    $Path = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'
}

# --- PATH first: this half is useful even when there is no .env ---------------------
$machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$user    = [Environment]::GetEnvironmentVariable('Path', 'User')
$env:Path = ($machine, $user | Where-Object { $_ }) -join ';'
Write-Host "PATH refreshed from the registry (newly installed CLIs are now visible)."

if (-not (Test-Path $Path)) {
    Write-Warning "No .env at $Path - PATH was refreshed, but no variables were loaded."
    return
}

$loaded  = New-Object System.Collections.Generic.List[string]
$skipped = New-Object System.Collections.Generic.List[string]

foreach ($raw in (Get-Content -LiteralPath $Path -Encoding UTF8)) {
    $line = $raw.Trim()
    if ($line -eq '' -or $line.StartsWith('#') -or ($line -notmatch '=')) { continue }

    $key   = $line.Substring(0, $line.IndexOf('=')).Trim()
    $value = $line.Substring($line.IndexOf('=') + 1).Trim()
    if ($key.StartsWith('export ')) { $key = $key.Substring(7).Trim() }
    if ($key -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { continue }

    # Strip matched surrounding quotes only. A '#' inside a value is part of the value,
    # never a comment - truncating a password there fails in a way nobody can diagnose.
    if ($value.Length -ge 2 -and $value[0] -eq $value[-1] -and ($value[0] -eq '"' -or $value[0] -eq "'")) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    if (-not $Force -and (Test-Path "env:$key")) { $skipped.Add($key); continue }
    Set-Item -Path "env:$key" -Value $value
    $loaded.Add($key)
}

# Names only. Never the values - this output gets pasted into chats and tickets.
Write-Host ("Loaded {0} variable(s) from .env: {1}" -f $loaded.Count, ($loaded -join ', '))
if ($skipped.Count -gt 0) {
    Write-Host ("Kept {0} already set in this session: {1}" -f $skipped.Count, ($skipped -join ', '))
    Write-Host "Use -Force to let the file win."
}
