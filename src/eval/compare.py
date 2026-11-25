"""여러 실행 결과를 한 표로 합친다.

예) python compare.py runs/yoloe_yoloe-26s-seg_name runs/yoloworld_yolov8s-worldv2_name
    (인자 없이 실행하면 runs/ 아래 전부)
결과: comparison.csv, comparison.md
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
runs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in (HERE / "runs").iterdir() if (p / "all").exists())

cols = ["run", "split", "images", "ms/img(mean±std)", "FPS", "mAP50(all)", "mAP50(>=5GT)", "precision", "recall", "recall_hand", "recall_free",
        "FP/img", "FP_neg_img", "wrong_cls"]
rows = []
classes = None
for r in runs:
    for split in ["all", "val", "test"]:
        f = r / split / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        classes = classes or list(m["per_class"])
        fmt = lambda v: "" if v != v else f"{v:.3f}"
        sp = m["speed"]
        ms = sp.get("ms_per_image_mean", float("nan"))
        sd = sp.get("ms_per_image_std", float("nan"))
        row = [r.name, split, m["images"], "" if ms != ms else (f"{ms:.1f}±{sd:.1f}" if sd == sd else f"{ms:.1f}"),
               "" if ms != ms else f"{1000 / ms:.1f}",
               fmt(m["mAP50_all_classes"]), fmt(m["mAP50_classes_ge5gt"]), fmt(m["precision@op_overall"]),
               fmt(m["recall@op_overall"]),
               fmt(m["recall@op_hand_overlap"]["recall@op"]), fmt(m["recall@op_free"]["recall@op"]),
               f"{m['fp@op_per_image']:.2f}", m["fp@op_on_negative_images"]["fp"], m["wrong_class@op"]]
        row += [fmt(m["per_class"][c]["AP50"]) for c in classes]
        row += [fmt(m["per_class"][c]["precision@op"]) for c in classes]
        row += [fmt(m["per_class"][c]["recall@op"]) for c in classes]
        rows.append(row)

header = cols + [f"{k}_{c}" for k in ["AP50", "P", "R"] for c in (classes or [])]
with open(HERE / "comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
(HERE / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
