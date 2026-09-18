# 历史取证脚本。2026-09-18 工作区重构后，上游资料位于与基线并列的 _archive/。
# 路径按需调整；本脚本不是正常开发或实机运行的依赖，产物已归档在 docs/history/。
$ErrorActionPreference = 'Stop'

$workspace = 'D:\VLA\_archive'
$dll = 'D:\VLA\_archive\dummy_v2\upstream\dummy-v2-ren-archive\3.Software\DummyStudio\DummyStudio_Data\Managed\Assembly-CSharp.dll'
if (-not $dll) {
    throw "Could not locate DummyStudio Assembly-CSharp.dll under $workspace"
}
$managed = Split-Path -LiteralPath $dll

[System.AppDomain]::CurrentDomain.add_AssemblyResolve({
    param($sender, $args)
    $name = ($args.Name -split ',')[0] + '.dll'
    $path = Join-Path $managed $name
    if (Test-Path -LiteralPath $path) {
        return [System.Reflection.Assembly]::LoadFrom($path)
    }
    return $null
}) | Out-Null

$one = @{}
$two = @{}
foreach ($field in [System.Reflection.Emit.OpCodes].GetFields([System.Reflection.BindingFlags]'Public,Static')) {
    $op = [System.Reflection.Emit.OpCode]$field.GetValue($null)
    $value = [int]$op.Value
    if (($value -band 0xff00) -eq 0xfe00) {
        $two[$value -band 0xff] = $op
    } else {
        $one[$value] = $op
    }
}

function Format-Token($module, [int]$token) {
    try { return $module.ResolveMember($token).ToString() } catch {}
    try { return '"' + $module.ResolveString($token) + '"' } catch {}
    return ('0x{0:x8}' -f $token)
}

function Read-I4($bytes, [int]$offset) {
    return [BitConverter]::ToInt32($bytes, $offset)
}

function Read-I8($bytes, [int]$offset) {
    return [BitConverter]::ToInt64($bytes, $offset)
}

function Disasm-Method($method) {
    $body = $method.GetMethodBody()
    if ($null -eq $body) {
        "  <no method body>"
        return
    }

    $bytes = $body.GetILAsByteArray()
    $module = $method.Module
    $i = 0
    while ($i -lt $bytes.Length) {
        $offset = $i
        $b = [int]$bytes[$i]
        $i += 1
        if ($b -eq 0xfe) {
            $op = $two[[int]$bytes[$i]]
            $i += 1
        } else {
            $op = $one[$b]
        }

        $operand = ''
        switch ($op.OperandType.ToString()) {
            'InlineNone' {}
            'ShortInlineI' { $signed = [int]$bytes[$i]; if ($signed -ge 128) { $signed -= 256 }; $operand = $signed.ToString(); $i += 1 }
            'InlineI' { $operand = (Read-I4 $bytes $i).ToString(); $i += 4 }
            'InlineI8' { $operand = (Read-I8 $bytes $i).ToString(); $i += 8 }
            'ShortInlineR' { $operand = ([BitConverter]::ToSingle($bytes, $i)).ToString('R', [Globalization.CultureInfo]::InvariantCulture); $i += 4 }
            'InlineR' { $operand = ([BitConverter]::ToDouble($bytes, $i)).ToString('R', [Globalization.CultureInfo]::InvariantCulture); $i += 8 }
            'ShortInlineVar' { $operand = ([int]$bytes[$i]).ToString(); $i += 1 }
            'InlineVar' { $operand = ([BitConverter]::ToUInt16($bytes, $i)).ToString(); $i += 2 }
            'ShortInlineBrTarget' {
                $delta = [int]$bytes[$i]
                if ($delta -ge 128) { $delta -= 256 }
                $i += 1
                $operand = ('IL_{0:x4}' -f ($i + $delta))
            }
            'InlineBrTarget' { $delta = Read-I4 $bytes $i; $i += 4; $operand = ('IL_{0:x4}' -f ($i + $delta)) }
            'InlineSwitch' {
                $count = Read-I4 $bytes $i
                $i += 4
                $base = $i + 4 * $count
                $targets = @()
                for ($j = 0; $j -lt $count; $j++) {
                    $delta = Read-I4 $bytes $i
                    $i += 4
                    $targets += ('IL_{0:x4}' -f ($base + $delta))
                }
                $operand = [string]::Join(', ', $targets)
            }
            default {
                $token = Read-I4 $bytes $i
                $i += 4
                $operand = Format-Token $module $token
            }
        }

        if ($operand -eq '') {
            '  IL_{0:x4}: {1}' -f $offset, $op.Name
        } else {
            '  IL_{0:x4}: {1,-12} {2}' -f $offset, $op.Name, $operand
        }
    }
}

$asm = [System.Reflection.Assembly]::LoadFrom($dll)
try { $loadedTypes = $asm.GetTypes() } catch [System.Reflection.ReflectionTypeLoadException] { $loadedTypes = $_.Exception.Types; foreach($failure in $_.Exception.LoaderExceptions) { "TYPE_LOAD_ERROR $failure" } }
$types = $loadedTypes | Where-Object { $null -ne $_ -and $_.FullName -like 'SerialPortUtility.DebugConsole*' } | ForEach-Object { $_.FullName }
$flags = [System.Reflection.BindingFlags]'Public,NonPublic,Instance,Static,DeclaredOnly'

foreach ($typeName in $types) {
    $type = $asm.GetType($typeName)
    if ($null -eq $type) {
        "TYPE NOT FOUND $typeName"
        continue
    }

    "TYPE $($type.FullName)"
    foreach ($field in $type.GetFields($flags)) {
        "FIELD $($field.FieldType.FullName) $($field.Name)"
    }
    foreach ($method in $type.GetMethods($flags)) {
        "METHOD $($method.Name)"
        Disasm-Method $method
    }
}
