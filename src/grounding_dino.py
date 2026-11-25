
from pathlib import Path
from time import perf_counter
import json

from PIL import Image, ImageDraw
from transformers import pipeline


# 1. 프로젝트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "images" / "test01.jpg"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# 2. 탐지할 물체 지정
LABELS = [
    "white wireless earbuds charging case"
]



# 3. Grounding DINO 모델 로드
print("Grounding DINO 모델 로딩 중...")

detector = pipeline(
    task="zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-base",
    device="cpu"
)


# 4. 이미지 불러오기
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"이미지를 찾을 수 없습니다: {IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print("현재 탐지 라벨:", LABELS)

results = detector(
    image,
    candidate_labels=LABELS,
    threshold=0.3
)

# 5. 객체 탐지
print("객체 탐지 시작!")

start = perf_counter()

results = detector(
    image,
    candidate_labels=LABELS,
    threshold=0.3
)

elapsed = perf_counter() - start


# 6. Bounding Box 시각화
output_image = image.copy()
draw = ImageDraw.Draw(output_image)

detections = []

for result in results:
    box = result["box"]

    label = str(result["label"])
    score = float(result["score"])

    bbox = [
        int(box["xmin"]),
        int(box["ymin"]),
        int(box["xmax"]),
        int(box["ymax"])
    ]

    draw.rectangle(
        bbox,
        outline="red",
        width=3
    )

    draw.text(
        (bbox[0], max(0, bbox[1] - 18)),
        f"{label}: {score:.2f}",
        fill="red"
    )

    detections.append({
        "label": label,
        "confidence": score,
        "bbox": bbox
    })

    print(
        f"탐지: {label}, "
        f"신뢰도: {score:.3f}, "
        f"좌표: {bbox}"
    )


# 7. 결과 이미지 저장
image_output_path = RESULT_DIR / "grounding_dino.jpg"

output_image.save(image_output_path)


# 8. JSON 저장
json_output_path = RESULT_DIR / "grounding_dino.json"

output = {
    "model": "IDEA-Research/grounding-dino-base",
    "image": IMAGE_PATH.name,
    "inference_seconds": elapsed,
    "detections": detections
}

with open(
    json_output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


# 9. 결과 출력
print("-----------------------------")
print(f"탐지된 물체 수: {len(detections)}")
print(f"추론 시간: {elapsed:.3f}초")
print(f"이미지 저장: {image_output_path}")
print(f"JSON 저장: {json_output_path}")
print("Grounding DINO 테스트 완료!")
