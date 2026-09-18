# Compatibility entrypoint. See tools/diagnostics/inspect_studio_native.ps1.
& (Join-Path $PSScriptRoot "diagnostics/inspect_studio_native.ps1") @args
