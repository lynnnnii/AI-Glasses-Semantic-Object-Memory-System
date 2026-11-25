"""직접 만든 추론 코드의 예측 결과를 공통 규칙으로 채점한다.

예측 폴더 구조 (이미지 파일 이름과 같은 stem 의 txt):
  <pred_dir>/val/<image_stem>.txt
  <pred_dir>/test/<image_stem>.txt
각 줄: class_id cx cy w h conf   (정규화 0~1, class_id 는 아래 순서)
  0 bottle, 1 bucket, 2 faucet, 3 mug, 4 pliers, 5 smartphone
conf 는 0.001 정도로 낮게 두고 모든 예측을 저장할 것 (AP 계산에 필요).

usage: python evaluate.py <pred_dir> [--name my_run] [--speed-json runs/speed_xxx.json]
       (또는 --ms-per-image 12.3 --ms-std 1.1)
"""
import argparse
import json
from pathlib import Path

from evalcore import evaluate, save_report

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("pred_dir")
ap.add_argument("--data", default=str(HERE.parents[1] / "data" / "egograsp_eval"))
ap.add_argument("--name", default=None)
ap.add_argument("--op-conf", type=float, default=0.25)
ap.add_argument("--ms-per-image", type=float, default=float("nan"), help="직접 잰 이미지당 평균 처리 시간(ms)")
ap.add_argument("--ms-std", type=float, default=float("nan"), help="처리 시간 표준편차(ms)")
ap.add_argument("--speed-json", default=None, help="speed.py 결과 json (있으면 ms 인자 대신 사용)")
args = ap.parse_args()

name = args.name or Path(args.pred_dir).name
out = HERE / "runs" / name
speed = json.loads(Path(args.speed_json).read_text()) if args.speed_json else \
    {"ms_per_image_mean": args.ms_per_image, "ms_per_image_std": args.ms_std}
info = {"model": name, "prompt_set": "custom", "speed": speed}
for split_name, splits in [("all", ["val", "test"]), ("val", ["val"]), ("test", ["test"])]:
    save_report(evaluate(args.data, args.pred_dir, splits, op_conf=args.op_conf), out / split_name, extra=info)

m = json.loads((out / "all" / "metrics.json").read_text(encoding="utf-8"))
print(f"[{name}] mAP50(>=5GT)={m['mAP50_classes_ge5gt']:.3f}  P={m['precision@op_overall']:.3f}  R={m['recall@op_overall']:.3f}  "
      f"recall_hand={m['recall@op_hand_overlap']['recall@op']:.3f}  FP/img={m['fp@op_per_image']:.2f}")
print(f"saved -> {out}  (compare.py 로 다른 결과와 합칠 수 있음)")
