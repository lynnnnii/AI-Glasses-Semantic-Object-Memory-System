
from pathlib import Path
from time import perf_counter
import json

import cv2
from ultralytics import YOLOWorld


# 1. 프로젝트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results" / "yolo_world"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# 2. 테스트 이미지
IMAGE_FILES = [
    "test01.jpg",
    "test02.jpg"
]


# 3. 탐지할 물체
LABELS = [
    "wireless earbuds charging case",
    "over-ear headphones"
]

CONF_THRESHOLD = 0.1
IMAGE_SIZE = 960


# 4. YOLO-World 모델 로드
print("YOLO-World 모델 로딩 중...")

model = YOLOWorld("yolov8s-worldv2.pt")

# 텍스트 프롬프트 설정
model.set_classes(LABELS)

print("탐지 대상:", LABELS)


# 5. 첫 추론 워밍업
# 모델 초기 준비 시간을 측정에서 제외
warmup_image = None

for filename in IMAGE_FILES:
    candidate = IMAGE_DIR / filename
    if candidate.exists():
        warmup_image = candidate
        break

if warmup_image is None:
    raise FileNotFoundError("테스트 이미지를 찾을 수 없습니다.")

print("모델 워밍업 중...")

model.predict(
    source=str(warmup_image),
    conf=CONF_THRESHOLD,
    imgsz=IMAGE_SIZE,
    device="cpu",
    verbose=False
)


# 6. 이미지별 탐지
for filename in IMAGE_FILES:
    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        print(f"이미지 없음: {image_path}")
        continue

    print(f"\n객체 탐지 시작: {filename}")

    start = perf_counter()

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )

    elapsed = perf_counter() - start

    result = results[0]


    # 7. Bounding Box 시각화
    annotated_image = result.plot()

    output_image_path = (
        RESULT_DIR / f"{image_path.stem}_detected.jpg"
    )

    cv2.imwrite(
        str(output_image_path),
        annotated_image
    )


    # 8. 탐지 결과 저장
    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = result.names[class_id]
        confidence = float(box.conf.item())

        bbox = [
            round(float(value), 2)
            for value in box.xyxy[0].tolist()
        ]

        detections.append({
            "label": label,
            "confidence": confidence,
            "bbox": bbox
        })

        print(
            f"탐지: {label}, "
            f"신뢰도: {confidence:.3f}, "
            f"좌표: {bbox}"
        )


    # 9. JSON 저장
    output = {
        "model": "yolov8s-worldv2.pt",
        "image": filename,
        "prompts": LABELS,
        "threshold": CONF_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "device": "cpu",
        "inference_seconds": elapsed,
        "detections": detections
    }

    output_json_path = (
        RESULT_DIR / f"{image_path.stem}_detected.json"
    )

    with open(
        output_json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("-----------------------------")
    print(f"탐지 물체 수: {len(detections)}")
    print(f"추론 시간: {elapsed:.3f}초")
    print(f"FPS 환산: {1 / elapsed:.3f}")
    print(f"결과 이미지: {output_image_path}")
    print(f"JSON: {output_json_path}")

print("\nYOLO-World 테스트 완료!")
