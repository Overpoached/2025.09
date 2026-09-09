<#
.SYNOPSIS
    Lock, reset or inspect GPU clocks so benchmark runs are comparable.

.DESCRIPTION
    A GPU left to its own devices boosts and drops clocks based on temperature,
    power and whatever else is on screen. Two runs of the same kernel minutes
    apart then differ for reasons that have nothing to do with the kernel.
    Locking the clocks trades peak throughput for reproducibility, which is the
    right trade for every number in BENCHMARKS.md.

    Requires an elevated prompt: clock control is a privileged operation.

    Caveats on this hardware, worth knowing before trusting the output:
      * -lgc (graphics/SM clock) generally works on GeForce.
      * -lmc (memory clock) is usually rejected on consumer cards. Since this
        project is bandwidth-bound, that matters, and it is why the benchmark
        records the observed memory clock either side of every run instead of
        assuming it held.
      * Persistence mode does not exist under Windows WDDM.

    Record whatever this reports in BENCHMARKS.md. "Could not lock the memory
    clock, observed variance was X%" is a perfectly good entry; a number with no
    stated clock state is not.

.EXAMPLE
    .\lock_clocks.ps1 -Status
.EXAMPLE
    .\lock_clocks.ps1 -Lock -GraphicsClock 2100
.EXAMPLE
    .\lock_clocks.ps1 -Reset
#>
[CmdletBinding(DefaultParameterSetName = 'Status')]
param(
    [Parameter(ParameterSetName = 'Status')]
    [switch]$Status,

    [Parameter(ParameterSetName = 'Lock')]
    [switch]$Lock,

    # Chosen well below peak boost so the clock is sustainable indefinitely and
    # the GPU never has to drop out of it mid-run.
    [Parameter(ParameterSetName = 'Lock')]
    [int]$GraphicsClock = 2100,

    [Parameter(ParameterSetName = 'Lock')]
    [int]$MemoryClock = 0,

    [Parameter(ParameterSetName = 'Reset')]
    [switch]$Reset
)

$ErrorActionPreference = 'Stop'

function Test-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Show-State {
    Write-Host "`nCurrent state:" -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv
}

function Invoke-Smi([string[]]$SmiArgs, [string]$Description) {
    Write-Host "`n$Description" -ForegroundColor Cyan
    Write-Host "  nvidia-smi $($SmiArgs -join ' ')" -ForegroundColor DarkGray
    $output = & nvidia-smi @SmiArgs 2>&1
    $exit = $LASTEXITCODE
    $output | ForEach-Object { Write-Host "  $_" }
    if ($exit -ne 0) {
        Write-Host "  -> FAILED (exit $exit). Record this in BENCHMARKS.md rather than ignoring it." -ForegroundColor Yellow
        return $false
    }
    return $true
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi not found on PATH."
}

switch ($PSCmdlet.ParameterSetName) {
    'Lock' {
        if (-not (Test-Elevated)) {
            throw "Locking clocks requires an elevated prompt. Re-run this script as Administrator."
        }

        Write-Host "Supported graphics clocks (a lock target must be one of these):" -ForegroundColor Cyan
        nvidia-smi --query-supported-clocks=gr --format=csv,noheader | Select-Object -First 12

        [void](Invoke-Smi @('-lgc', "$GraphicsClock,$GraphicsClock") `
            "Locking SM clock to $GraphicsClock MHz")

        if ($MemoryClock -gt 0) {
            [void](Invoke-Smi @('-lmc', "$MemoryClock,$MemoryClock") `
                "Locking memory clock to $MemoryClock MHz (often unsupported on GeForce)")
        }

        Show-State
        Write-Host "`nRun the benchmark now, then reset with: .\lock_clocks.ps1 -Reset" -ForegroundColor Green
    }

    'Reset' {
        if (-not (Test-Elevated)) {
            throw "Resetting clocks requires an elevated prompt. Re-run this script as Administrator."
        }
        [void](Invoke-Smi @('-rgc') "Resetting SM clock to default behaviour")
        [void](Invoke-Smi @('-rmc') "Resetting memory clock to default behaviour")
        Show-State
    }

    default {
        Show-State
        Write-Host "`nSupported graphics clocks:" -ForegroundColor Cyan
        nvidia-smi --query-supported-clocks=gr --format=csv,noheader | Select-Object -First 12
        Write-Host "`nElevated: $(Test-Elevated)"
        Write-Host "To lock:  .\lock_clocks.ps1 -Lock -GraphicsClock 2100   (elevated prompt)"
    }
}
