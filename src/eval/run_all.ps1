# 두 모델 x 3개 프롬프트 세트를 같은 PC(CPU)에서 순차 실행 (속도 측정이 서로 간섭하지 않도록)
# 기본 규칙: 워밍업 10회 제외, 85장 x 5회
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Set-Location $PSScriptRoot
$py = "..\..\.venv\Scripts\python.exe"
$jobs = @(
  @("yoloe", "yoloe-26s-seg.pt", "name", 5),
  @("yoloworld", "yolov8s-worldv2.pt", "name", 5),
  @("yoloworld", "yolov8s-worldv2.pt", "desc", 5),
  @("yoloworld", "yolov8s-worldv2.pt", "multi", 5),
  @("yoloe", "yoloe-26s-seg.pt", "desc", 5),
  @("yoloe", "yoloe-26s-seg.pt", "multi", 5)
  # Grounding DINO 는 AI 안경(엣지) 배포 불가로 비교에서 제외 (CPU ~28초/장 vs YOLOE-26s ~0.27초/장)
  # 자동 라벨링 용도로 볼 때: @("gdino", "IDEA-Research/grounding-dino-base", "name", 1)
)
foreach ($j in $jobs) {
  Write-Output "===== $(Get-Date -Format HH:mm:ss) START $($j -join ' ')"
  & $py run_model.py --model $j[0] --weights $j[1] --prompt-set $j[2] --repeats $j[3] --device cpu --save-vis 2>&1 |
    Where-Object { $_ -notmatch "Downloading|FutureWarning|warnings.warn" } | ForEach-Object { "$_" }
  Write-Output "===== $(Get-Date -Format HH:mm:ss) END exit=$LASTEXITCODE"
}
Write-Output "ALL DONE"
