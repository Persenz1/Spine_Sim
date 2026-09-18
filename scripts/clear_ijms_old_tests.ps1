# User-requested manual cleanup; never touches the production directory.
$ErrorActionPreference = 'Stop'
$expected = 'E:\TestData\IJMS\old_tests_pending_delete'
if (-not (Test-Path -LiteralPath $expected)) { Write-Output 'No old test directory remains.'; return }
$resolved = (Resolve-Path -LiteralPath $expected).Path
if (-not [string]::Equals($resolved, $expected, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Cleanup target does not match the named old-test directory.'
}
$entry = Get-Item -LiteralPath $resolved -Force
if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing a reparse-point target.' }
$links = @(Get-ChildItem -LiteralPath $resolved -Recurse -Force |
    Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
if ($links.Count) { throw 'Refusing recursive deletion across a nested reparse point.' }
Write-Output "Removing old tests only: $resolved"
Remove-Item -LiteralPath $resolved -Recurse -Force
