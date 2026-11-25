# EgoObjects val: 두 모델 x 프롬프트 세트 3개를 같은 PC(CPU)에서 순차 실행
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$py = "..\..\.venv\Scripts\python.exe"
$jobs = @(
  @("yoloe", "yoloe-26s-seg.pt", "name"),
  @("yoloworld", "yolov8s-worldv2.pt", "name"),
  @("yoloe", "yoloe-26s-seg.pt", "desc"),
  @("yoloworld", "yolov8s-worldv2.pt", "desc"),
  @("yoloe", "yoloe-26s-seg.pt", "multi"),
  @("yoloworld", "yolov8s-worldv2.pt", "multi")
)
foreach ($j in $jobs) {
  Write-Output "===== $(Get-Date -Format HH:mm:ss) START $($j -join ' ')"
  & $py ego_run.py --model $j[0] --weights $j[1] --prompt-set $j[2] --device cpu 2>&1 |
    Where-Object { $_ -notmatch "Downloading|FutureWarning|warnings.warn" } | ForEach-Object { "$_" }
  Write-Output "===== $(Get-Date -Format HH:mm:ss) END exit=$LASTEXITCODE"
}
& $py ego_compare.py
Write-Output "ALL DONE"
