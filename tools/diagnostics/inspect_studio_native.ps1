$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Add-Type -Path "$root\.diagnostics\cecil\package\lib\net40\Mono.Cecil.dll"
$a=[Mono.Cecil.AssemblyDefinition]::ReadAssembly("$root\.diagnostics\DummyStudioCapture\DummyStudio_Data\Managed\Assembly-CSharp.dll")
$t=$a.MainModule.Types | Where-Object FullName -eq 'SerialPortUtility.SerialPortUtilityPro'
foreach($n in $t.NestedTypes) {
  if($n.Name -in @('SpapConfig','ParityEnum','StopBitEnum','DataBitEnum')) {
    "TYPE $($n.FullName) layout=$($n.Attributes) pack=$($n.PackingSize)"
    $n.Fields | ForEach-Object { "FIELD $($_.FieldType) $($_.Name) value=$($_.Constant)" }
  }
}
foreach($m in $t.Methods) {
  if($m.IsPInvokeImpl -and $m.Name -in @('spapOpenUSB','spapClose','spapWrite','spapReadData','spapReadDataAvailable','spapGetDTR','spapGetRTS')) {
    "METHOD $($m.FullName) module=$($m.PInvokeInfo.Module.Name) flags=$($m.PInvokeInfo.Attributes) entry=$($m.PInvokeInfo.EntryPoint)"
    $m.Parameters | ForEach-Object { " PARAM $($_.Name) $($_.ParameterType) marshal=$($_.MarshalInfo)" }
  }
}
$a.Dispose()
