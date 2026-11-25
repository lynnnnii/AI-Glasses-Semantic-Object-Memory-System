# 제로샷 객체 탐지 모델 비교 (EgoGrasp)

YOLOE-26s와 YOLO-World v2(Small)를 같은 데이터, 같은 프롬프트, 같은 채점 코드로 비교한 결과입니다.
Grounding DINO는 같은 CPU에서 이미지당 약 28초(YOLOE-26s는 약 0.27초)로 AI 안경에 올릴 수 없어 비교에서 제외했습니다.

## 결론

- 같은 프롬프트로 비교하면 mAP50(GT 5개 이상 클래스) 차이는 name 0.007, multi 0.010으로 팀 기준(0.15)에 못 미칩니다.
- desc에서만 YOLOE가 0.187 앞서지만, 클립 단위 부트스트랩 95% 구간(−0.05 ~ +0.21)이 0을 포함합니다.
- 평가 데이터가 10개 클립뿐이라 제로샷 정확도로는 두 모델을 가를 수 없습니다. 다음 기준(속도, 엣지 배포, YOLO26 fine-tuning 연속성, 시각 프롬프트)으로 결정합니다.

## 결과 (split=all, 85장)

| 프롬프트 | 모델 | mAP50 (GT≥5) | Precision | Recall | recall_hand | 다른 클래스로 부름 | ms/장 (CPU) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| name | YOLOE-26s | 0.614 | 0.721 | 0.662 | 0.681 | 8 | 270 ± 51 |
| name | YOLO-World v2 | 0.621 | 0.676 | 0.676 | 0.702 | 11 | 317 ± 132 |
| desc | YOLOE-26s | 0.798 | 0.764 | 0.568 | 0.532 | 0 | 299 ± 41 |
| desc | YOLO-World v2 | 0.611 | 0.758 | 0.635 | 0.574 | 10 | 253 ± 49 |
| multi | YOLOE-26s | 0.673 | 0.771 | 0.730 | 0.766 | 5 | 350 ± 116 |
| multi | YOLO-World v2 | 0.683 | 0.705 | 0.743 | 0.745 | 11 | 309 ± 84 |

전체 지표와 클래스별 AP/P/R은 `comparison.md`, `comparison.csv`, `runs/<run>/{all,val,test}/`에 있습니다.
속도는 Intel Core Ultra 5 125H CPU(GPU 없음) 기준이며, 팀이 정한 PC에서 다시 측정해야 합니다.

## 평가 조건

- 데이터: EgoGrasp 샘플을 정리한 85장 (val 24 / test 61, 10개 클립), 물체 74개 (손과 겹침 47 / 안 겹침 27), 빈 이미지 16장
- 클래스: bottle, bucket, faucet, mug, pliers, smartphone (pliers는 GT 1개라 주 지표에서 제외)
- 추론: imgsz 640, batch 1, conf 0.001까지 저장, CPU에서 한 번에 하나씩 실행
- 채점(`evalcore.py`): IoU 0.5 greedy 매칭, COCO 101점 AP, ignore 박스 제외, 동의어 프롬프트는 클래스별 NMS, P/R은 conf 0.25 기준
- 속도(`speed.py`): 워밍업 10회 제외, 85장 × 5회, 전처리+추론+후처리 벽시계 시간

## 실행 방법

평가 데이터는 재배포 금지(내부 평가용)라 저장소에 없습니다. 팀 공유 폴더의 `egograsp_eval/`을 저장소 루트의 `data/egograsp_eval/`에 두고 실행합니다.

```powershell
cd src/eval
python run_model.py --model yoloe --weights yoloe-26s-seg.pt --prompt-set name --device cpu --save-vis
python run_model.py --model yoloworld --weights yolov8s-worldv2.pt --prompt-set name --device cpu --save-vis
python compare.py
```

- `run_all.ps1`: 두 모델 × 프롬프트 세트 3개를 순차 실행
- `evaluate.py`: 다른 추론 코드로 만든 예측 txt(`cls cx cy w h conf`, 정규화)를 같은 규칙으로 채점
- `gdino.py`: Grounding DINO를 같은 파이프라인에 붙이는 어댑터 (`--model gdino --weights IDEA-Research/grounding-dino-base`), 자동 라벨링 검토용
- `vis/`(예측 그림)와 `preds/`(예측 txt)는 데이터 이미지 정보가 들어 있어 커밋하지 않습니다.
