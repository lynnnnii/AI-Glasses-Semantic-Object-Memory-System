"""YOLOE-26 / YOLO-World 를 같은 조건으로 돌려 예측을 저장하고 채점한다.

예)
  python run_model.py --model yoloe --weights yoloe-26s-seg.pt --prompt-set name
  python run_model.py --model yoloworld --weights yolov8s-worldv2.pt --prompt-set name
  python run_model.py --model gdino --weights IDEA-Research/grounding-dino-base --prompt-set name

결과: runs/<model>_<weights>_<prompt-set>/
  preds/{val,test}/*.txt   예측 (cls cx cy w h conf, 정규화)
  metrics.json, per_class.csv
  vis/                      --save-vis 일 때 op_conf 이상 예측을 그린 이미지
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evalcore import NAMES, evaluate, save_report
from speed import load_model, measure

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["yoloe", "yoloworld", "gdino"])
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default=str(HERE.parents[1] / "data" / "egograsp_eval"))
ap.add_argument("--prompt-set", default="name")
ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--op-conf", type=float, default=0.25, help="Precision/Recall/FP를 잴 기준 confidence")
ap.add_argument("--device", default=None, help="예: cpu, 0, mps")
ap.add_argument("--save-vis", action="store_true")
ap.add_argument("--repeats", type=int, default=5, help="속도 측정 반복 횟수 (0이면 측정 생략)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

data = Path(args.data)
prompt_cfg = json.loads(Path(args.prompts).read_text(encoding="utf-8"))[args.prompt_set]
texts, text2cls = [], []
for c, name in enumerate(NAMES):
    for t in prompt_cfg[name]:
        texts.append(t)
        text2cls.append(c)

model = load_model(args.model, args.weights, texts, args.device)

run_name = args.out or f"{args.model}_{Path(args.weights).stem}_{args.prompt_set}"
out = HERE / "runs" / run_name
pred_root = out / "preds"

images = [(s, p) for s in ["val", "test"] for p in sorted((data / s / "images").glob("*.jpg"))]
kw = dict(conf=0.001, imgsz=args.imgsz, verbose=False)
if args.device is not None:
    kw["device"] = args.device

for split, img in images:
    r = model.predict(str(img), **kw)[0]
    lines = []
    if r.boxes is not None and len(r.boxes):
        for (cx, cy, w, h), k, cf in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            lines.append(f"{text2cls[int(k)]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.5f}")
    d = pred_root / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{img.stem}.txt").write_text("\n".join(lines))

if args.repeats > 0:
    print(f"속도 측정 중 (워밍업 후 {len(images)}장 x {args.repeats}회)...")
    speed = measure(model, [p for _, p in images], args.imgsz, args.device, repeats=args.repeats)
else:
    speed = {"ms_per_image_mean": float("nan"), "ms_per_image_std": float("nan"), "fps": float("nan")}
info = {"model": args.model, "weights": args.weights, "prompt_set": args.prompt_set,
        "prompts": prompt_cfg, "speed": speed}

for name, splits in [("all", ["val", "test"]), ("val", ["val"]), ("test", ["test"])]:
    m = evaluate(data, pred_root, splits, op_conf=args.op_conf)
    save_report(m, out / name, extra=info)

m = json.loads((out / "all" / "metrics.json").read_text(encoding="utf-8"))
print(f"\n[{run_name}]  images={m['images']}  speed={speed['ms_per_image_mean']:.1f}±{speed['ms_per_image_std']:.1f} ms/img  ({speed['fps']:.1f} FPS)")
print(f"  mAP50 (all cls)       : {m['mAP50_all_classes']:.3f}")
print(f"  mAP50 (cls with >=5GT): {m['mAP50_classes_ge5gt']:.3f}")
print(f"  precision@{args.op_conf}     : {m['precision@op_overall']:.3f}")
print(f"  recall@{args.op_conf} overall : {m['recall@op_overall']:.3f}")
print(f"  recall hand-overlap   : {m['recall@op_hand_overlap']['recall@op']:.3f} (n={m['recall@op_hand_overlap']['n']})")
print(f"  recall free           : {m['recall@op_free']['recall@op']:.3f} (n={m['recall@op_free']['n']})")
print(f"  FP per image          : {m['fp@op_per_image']:.2f}   wrong-class: {m['wrong_class@op']}")
print(f"  saved -> {out}")

if args.save_vis:
    import cv2
    vis = out / "vis"
    vis.mkdir(exist_ok=True)
    for split, img in images:
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        for ln in (pred_root / split / f"{img.stem}.txt").read_text().split("\n"):
            if not ln.strip():
                continue
            c, cx, cy, bw, bh, cf = map(float, ln.split())
            if cf < args.op_conf:
                continue
            p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
            p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
            cv2.rectangle(im, p1, p2, (0, 0, 255), 4)
            cv2.putText(im, f"{NAMES[int(c)]} {cf:.2f}", (p1[0], max(p1[1] - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(vis / f"{split}_{img.stem.split('_jpg')[0]}.jpg"), im)
