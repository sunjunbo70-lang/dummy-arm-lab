# 历史取证脚本。2026-09-18 工作区重构后，上游资料位于与基线并列的 _archive/。
# 路径按需调整；本脚本不是正常开发或实机运行的依赖，产物已归档在 docs/history/。
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$source = 'D:\VLA\_archive\dummy_v2\upstream\dummy-v2-ren-archive\3.Software\DummyStudio'
$dest = Join-Path $root '.diagnostics\DummyStudioCapture'
if (Test-Path "$dest\DummyStudio_Data\Managed\RobotSerialTrace.dll") { throw 'Capture copy already patched; do not patch twice.' }
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Copy-Item -LiteralPath "$source\DummyStudio.exe" -Destination $dest
Get-ChildItem -LiteralPath $source | Where-Object Name -ne 'DummyStudio.exe' | Copy-Item -Destination $dest -Recurse
$managed = Join-Path $dest 'DummyStudio_Data\Managed'
$log = (Join-Path $root 'outputs\studio_wire_capture.tsv').Replace('"','""')
$cs = @"
using System;
using System.IO;
using System.Text;
public static class RobotSerialTrace {
  static readonly object Gate = new object();
  public static void Record(string direction, byte[] data, int count) {
    try {
      if (data == null || count <= 0 || count > data.Length) return;
      string line = DateTime.UtcNow.ToString("O") + "\t" + direction + "\t" + Convert.ToBase64String(data,0,count) + "\n";
      lock(Gate) { File.AppendAllText(@"$log", line, Encoding.UTF8); }
    } catch { }
  }
}
"@
Add-Type -TypeDefinition $cs -OutputAssembly "$managed\RobotSerialTrace.dll" -OutputType Library
Add-Type -Path "$root\.diagnostics\cecil\package\lib\net40\Mono.Cecil.dll"
$resolver = New-Object Mono.Cecil.DefaultAssemblyResolver
$resolver.AddSearchDirectory($managed)
$rp = New-Object Mono.Cecil.ReaderParameters
$rp.AssemblyResolver = $resolver
$path = "$managed\Assembly-CSharp.dll"
$originalHash = (Get-FileHash -LiteralPath $path).Hash
$asm = [Mono.Cecil.AssemblyDefinition]::ReadAssembly($path,$rp)
$helper = [Mono.Cecil.AssemblyDefinition]::ReadAssembly("$managed\RobotSerialTrace.dll",$rp)
$recordMethod = ($helper.MainModule.Types | Where-Object Name -eq 'RobotSerialTrace').Methods | Where-Object Name -eq 'Record'
$record = $asm.MainModule.ImportReference($recordMethod)
$type = $asm.MainModule.Types | Where-Object FullName -eq 'SerialPortUtility.SerialPortUtilityPro'
$write = $type.Methods | Where-Object { $_.Name -eq 'Write' -and $_.Parameters.Count -eq 1 -and $_.Parameters[0].ParameterType.FullName -eq 'System.Byte[]' }
$read = $type.Methods | Where-Object Name -eq 'ReadUpdate'
function Insert-Trace($method,$anchor,$instructions) {
  $il = $method.Body.GetILProcessor()
  foreach ($instruction in $instructions) { $il.InsertAfter($anchor,$instruction); $anchor=$instruction }
  # Expand short branches so insertion cannot overflow their signed-byte displacement.
  foreach ($ins in $method.Body.Instructions) {
    if ($ins.OpCode.OperandType -eq [Mono.Cecil.Cil.OperandType]::ShortInlineBrTarget) {
      $longName=$ins.OpCode.Code.ToString() -replace '_S$',''
      $ins.OpCode=[Mono.Cecil.Cil.OpCodes].GetField($longName).GetValue($null)
    }
  }
}
$op=[Mono.Cecil.Cil.OpCodes]
$tx=@($write.Body.Instructions | Where-Object { $_.Operand -is [Mono.Cecil.MethodReference] -and $_.Operand.Name -eq 'spapWrite' })
$rx=@($read.Body.Instructions | Where-Object { $_.Operand -is [Mono.Cecil.MethodReference] -and $_.Operand.Name -eq 'spapReadData' })
if ($tx.Count -ne 1 -or $rx.Count -ne 1 -or $rx[0].Next.OpCode.Code.ToString() -ne 'Stloc_2') { throw 'Unexpected original IL' }
Insert-Trace $write $tx[0] @(
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldstr,'TX'),
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldarg_1),
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldarg_1),
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldlen),
  [Mono.Cecil.Cil.Instruction]::Create($op::Conv_I4),
  [Mono.Cecil.Cil.Instruction]::Create($op::Call,$record)
)
Insert-Trace $read $rx[0].Next @(
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldstr,'RX'),
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldloc_1),
  [Mono.Cecil.Cil.Instruction]::Create($op::Ldloc_2),
  [Mono.Cecil.Cil.Instruction]::Create($op::Call,$record)
)
$asm.Write("$path.patched")
$asm.Dispose(); $helper.Dispose()
Move-Item -LiteralPath "$path.patched" -Destination $path -Force
$check=[Mono.Cecil.AssemblyDefinition]::ReadAssembly($path,$rp)
$hooks=@(($check.MainModule.Types | Where-Object FullName -eq 'SerialPortUtility.SerialPortUtilityPro').Methods | ForEach-Object { if ($_.HasBody) { $_.Body.Instructions | Where-Object { $_.Operand -is [Mono.Cecil.MethodReference] -and $_.Operand.DeclaringType.Name -eq 'RobotSerialTrace' } } })
if ($hooks.Count -ne 2) { throw 'Hook verification failed' }
$check.Dispose()
if ((Get-FileHash "$source\DummyStudio_Data\Managed\Assembly-CSharp.dll").Hash -ne $originalHash) { throw 'Source hash mismatch' }
@{source=$source;destination=$dest;source_sha256=$originalHash;patched_sha256=(Get-FileHash $path).Hash;hooks=$hooks.Count;log=$log} | ConvertTo-Json | Set-Content "$root\outputs\studio_capture_manifest.json" -Encoding UTF8
Write-Output "Capture copy verified: $dest"
