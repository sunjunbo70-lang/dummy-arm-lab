param([ValidateSet('Inspect','Apply','Rollback')][string]$Mode='Inspect')
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$id='USB\VID_1209&PID_0D32&MI_02\6&14EC3AFE&1&0002'
$key="HKLM:\SYSTEM\CurrentControlSet\Enum\$id\Device Parameters"
$backupPath=Join-Path $root 'outputs\native_interface_registry_backup.json'
$resultPath=Join-Path $root 'outputs\native_interface_repair_result.json'
$result=@{mode=$Mode;instance=$id;time=(Get-Date -Format o);success=$false}
try {
    $device=Get-PnpDevice -InstanceId $id -PresentOnly
    $service=(Get-PnpDeviceProperty -InstanceId $id -KeyName DEVPKEY_Device_Service).Data
    if($service -ne 'WINUSB' -or $device.FriendlyName -ne 'REF 1.0 Native Interface') { throw 'Unexpected interface; refusing modification' }
    $current=Get-ItemProperty -LiteralPath $key
    $exists=$null -ne $current.PSObject.Properties['DeviceInterfaceGUIDs']
    $value=if($exists){@($current.DeviceInterfaceGUIDs)}else{@()}
    $result.current_guids=$value
    if($Mode -eq 'Inspect') { $result.success=$true; $result | ConvertTo-Json; return }
    $principal=[Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    if(-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator confirmation is required' }
    if($Mode -eq 'Apply') {
        if($exists -or $null -ne $current.PSObject.Properties['DeviceInterfaceGUID']) { throw 'An interface GUID value already exists; refusing to overwrite it' }
        if(Test-Path -LiteralPath $backupPath) { throw 'Backup already exists; inspect previous attempt before reapplying' }
        $newGuid='{'+[guid]::NewGuid().ToString().ToUpperInvariant()+'}'
        @{instance=$id;key=$key;original_exists=$false;added_guid=$newGuid;created=(Get-Date -Format o)} |
            ConvertTo-Json | Set-Content -LiteralPath $backupPath -Encoding UTF8
        New-ItemProperty -LiteralPath $key -Name DeviceInterfaceGUIDs -PropertyType MultiString -Value @($newGuid) | Out-Null
        $result.added_guid=$newGuid
    } else {
        $backup=Get-Content -LiteralPath $backupPath -Raw | ConvertFrom-Json
        if($backup.instance -ne $id -or $backup.original_exists) { throw 'Backup does not match this repair' }
        if($value.Count -ne 1 -or $value[0] -ne $backup.added_guid) { throw 'Registry value changed since repair; refusing rollback' }
        Remove-ItemProperty -LiteralPath $key -Name DeviceInterfaceGUIDs
    }
    # Restart only the native child interface (MI_02), not COM6 or the composite parent.
    $restartOutput=& "$env:WINDIR\System32\pnputil.exe" /restart-device $id 2>&1
    $result.restart_output=($restartOutput | Out-String)
    $result.restart_exit_code=$LASTEXITCODE
    if($LASTEXITCODE -ne 0) {
        if($Mode -eq 'Apply') {
            Remove-ItemProperty -LiteralPath $key -Name DeviceInterfaceGUIDs
            $result.registry_reverted=$true
        }
        throw 'Native interface restart failed; check result before any retry'
    }
    $result.success=$true
} catch {
    $result.error=$_.Exception.Message
    throw
} finally {
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultPath -Encoding UTF8
}
