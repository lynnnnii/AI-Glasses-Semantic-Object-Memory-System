"""EgoObjects 실행 결과를 표로 합치고, 같은 프롬프트의 YOLOE vs YOLO-World 차이를 촬영 세션 단위 부트스트랩으로 검정한다.

usage: python ego_compare.py            (runs_ego/ 아래 전부)
결과: runs_ego/comparison.md, runs_ego/comparison.csv, runs_ego/bootstrap.json
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ego_run import MAIN, NAMES, aggregate

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs_ego"
runs = sorted(p for p in ROOT.iterdir() if (p / "metrics.json").exists())
M = {p.name: json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in runs}

f = lambda v: "" if v != v else f"{v:.3f}"
cols = ["run", "ms/img", "mAP50(main4)", "mAP50(8cls)", "precision", "recall", "recall(main4)", "FP/img",
        "FP_on_absent", "wrong_cls"] + [f"AP50_{n}" for n in NAMES]
rows = []
for name, m in M.items():
    sp = m["speed"]
    rows.append([name, "" if sp["ms_per_image_mean"] != sp["ms_per_image_mean"] else f"{sp['ms_per_image_mean']:.0f}±{sp['ms_per_image_std']:.0f}",
                 f(m["mAP50_main4"]), f(m["mAP50_8cls"]), f(m["precision@op"]), f(m["recall@op"]),
                 f(m["recall@op_main4"]), f"{m['fp@op_per_image']:.2f}", m["fp@op_on_confirmed_absent"],
                 m["wrong_class@op"]] + [f(m["per_class"][n]["AP50"]) for n in NAMES])
with open(ROOT / "comparison.csv", "w", newline="", encoding="utf-8-sig") as fh:
    csv.writer(fh).writerows([cols] + rows)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]

# 세션 단위 부트스트랩: 같은 prompt set 의 yoloe vs yoloworld
boot = {}
for ps in ["name", "desc", "multi"]:
    a = next((p for p in runs if p.name.startswith("yoloe_") and p.name.endswith("_" + ps)), None)
    b = next((p for p in runs if p.name.startswith("yoloworld_") and p.name.endswith("_" + ps)), None)
    if not (a and b):
        continue
    pa = json.loads((a / "per_image.json").read_text(encoding="utf-8"))
    pb = json.loads((b / "per_image.json").read_text(encoding="utf-8"))
    by_g = defaultdict(list)
    for s, v in pa.items():
        by_g[v["group"]].append(s)
    groups = sorted(by_g)
    op = M[a.name]["op_conf"]
    rng = np.random.default_rng(0)
    d_map, d_rec = [], []
    for _ in range(1000):
        stems = [s for g in rng.choice(groups, len(groups)) for s in by_g[g]]
        ma, mb = aggregate(pa, stems, op), aggregate(pb, stems, op)
        d_map.append(ma["mAP50_main4"] - mb["mAP50_main4"])
        d_rec.append(ma["recall@op_main4"] - mb["recall@op_main4"])
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    boot[ps] = {"sessions": len(groups),
                "dmap_main4": M[a.name]["mAP50_main4"] - M[b.name]["mAP50_main4"], "dmap_main4_ci": q(d_map),
                "drecall_main4": M[a.name]["recall@op_main4"] - M[b.name]["recall@op_main4"], "drecall_main4_ci": q(d_rec)}
(ROOT / "bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
md += ["", "| prompt | ΔmAP50(main4) YOLOE−YW | 95% CI | Δrecall(main4) | 95% CI |", "|---|---|---|---|---|"]
for ps, v in boot.items():
    md.append(f"| {ps} | {v['dmap_main4']:+.3f} | {v['dmap_main4_ci'][0]:+.3f} ~ {v['dmap_main4_ci'][1]:+.3f} | "
              f"{v['drecall_main4']:+.3f} | {v['drecall_main4_ci'][0]:+.3f} ~ {v['drecall_main4_ci'][1]:+.3f} |")
(ROOT / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
