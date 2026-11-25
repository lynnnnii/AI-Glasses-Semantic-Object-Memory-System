# 제로샷 객체 탐지 모델 비교 (EgoGrasp, EgoObjects)

YOLOE-26s와 YOLO-World v2(Small)를 같은 데이터, 같은 프롬프트, 같은 채점 코드로 비교한 결과입니다.
Grounding DINO는 같은 CPU에서 이미지당 약 28초(YOLOE-26s는 약 0.27초)로 AI 안경에 올릴 수 없어 비교에서 제외했습니다.

## 종합 결론

- 우리 목적에 가까운 1인칭 데이터(EgoObjects, 1,623장)에서는 YOLOE-26s가 세 프롬프트 모두에서 앞섭니다. Recall 차이는 세션 단위 부트스트랩 95% 구간이 모두 0보다 크고, name 프롬프트의 Recall 차이(+0.156)는 팀 기준 0.15를 넘습니다.
- EgoGrasp(85장)에서는 차이가 우연과 구분되지 않았습니다.
- 어느 모델이든 제로샷 Recall이 최대 0.29이고 열쇠는 거의 못 잡으므로, 직접 찍은 영상으로 fine-tuning이 필요합니다.

## EgoObjects (1인칭, 손 지표 없음)

Meta EgoObjects val 중 아래 8개 클래스가 있거나 없다고 확인된 1,623장(촬영 세션 301개)입니다.
대상 클래스는 wallet(GT 236), house/car key(165), mobile phone(174), headphones(123, 에어팟 대신)이고, glasses, watch, remote control, phone charger는 참고용입니다.

| 프롬프트 | 모델 | mAP50 (대상 4) | mAP50 (8) | Precision | Recall (대상 4) | FP/장 | ms/장 (CPU) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| name | YOLOE-26s | 0.370 | 0.303 | 0.724 | 0.254 | 0.08 | 164 ± 27 |
| name | YOLO-World v2 | 0.319 | 0.275 | 0.699 | 0.097 | 0.05 | 135 ± 24 |
| desc | YOLOE-26s | 0.342 | 0.258 | 0.794 | 0.148 | 0.03 | 181 ± 30 |
| desc | YOLO-World v2 | 0.263 | 0.238 | 0.638 | 0.095 | 0.06 | 146 ± 31 |
| multi | YOLOE-26s | 0.380 | 0.299 | 0.700 | 0.291 | 0.10 | 187 ± 33 |
| multi | YOLO-World v2 | 0.352 | 0.290 | 0.660 | 0.156 | 0.08 | 147 ± 25 |

세션 단위 부트스트랩(1,000회) 95% 구간입니다.

| 프롬프트 | mAP50 차이 (YOLOE − YOLO-World) | 95% 구간 | Recall 차이 | 95% 구간 |
| --- | --- | --- | --- | --- |
| name | +0.051 | +0.012 ~ +0.094 | +0.156 | +0.104 ~ +0.217 |
| desc | +0.079 | +0.016 ~ +0.146 | +0.053 | +0.015 ~ +0.090 |
| multi | +0.028 | −0.013 ~ +0.070 | +0.135 | +0.080 ~ +0.190 |

- 지갑과 헤드폰은 YOLOE가 확실히 낫고, 휴대폰은 desc·multi에서 YOLO-World의 AP가 더 높습니다.
- 열쇠는 conf 0.25 기준 Recall이 YOLO-World 0~0.01, YOLOE 0.01~0.15로 둘 다 거의 못 잡습니다. watch와 phone charger도 둘 다 AP50 0.07 이하입니다.
- 속도는 YOLO-World가 약 15~20% 빠릅니다. YOLOE-26 seg 모델이 마스크까지 계산하기 때문일 수 있습니다.
- 채점은 LVIS federated 방식입니다. 클래스 c의 예측은 그 이미지에서 c가 있거나 없다고 확인된 경우에만 채점하고, 그 밖은 `evalcore.py`와 같습니다.
- 결과는 `runs_ego/comparison.md`, `runs_ego/bootstrap.json`, `runs_ego/<run>/{metrics.json, per_class.csv}`에 있습니다.

## EgoGrasp 결론

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
속도는 팀 비교용 PC의 Intel Core Ultra 5 125H CPU(GPU 없음) 기준입니다.

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

### EgoObjects

EgoObjects는 연구용 라이선스라 이미지와 주석을 저장소에 올리지 않습니다. 각자 라이선스에 동의하고 받은 val 이미지와 `annotations_val_subset.json`(8개 클래스만 추린 주석)을 저장소 루트의 `data/egoobjects_val/{images/, annotations_val_subset.json}`에 둡니다.

```powershell
cd src/eval
python ego_run.py --model yoloe --weights yoloe-26s-seg.pt --prompt-set name --device cpu
python ego_run.py --model yoloworld --weights yolov8s-worldv2.pt --prompt-set name --device cpu
python ego_compare.py
```

- `run_ego_all.ps1`: 두 모델 × 프롬프트 세트 3개를 순차 실행한 뒤 `ego_compare.py`까지 실행
- `prompts_ego.json`: EgoObjects 8개 클래스용 name, desc, multi 프롬프트
- `per_image.json`과 `preds/`는 이미지별 정보라 커밋하지 않습니다. 부트스트랩(`ego_compare.py`)을 다시 하려면 `ego_run.py`부터 실행해야 합니다.
