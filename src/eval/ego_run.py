"""EgoObjects val(1인칭, 스마트 안경 촬영)에서 YOLOE / YOLO-World 제로샷 비교. 손 지표 없음.

채점 규칙 (LVIS federated)
- EgoObjects는 이미지마다 일부 카테고리만 "있음(박스)" 또는 "없음(neg_category_ids)"으로 확인했다.
- 클래스 c 의 예측은 c 가 확인된 이미지(있음 또는 없음)에서만 채점하고, 확인 안 된 이미지의 c 예측은 버린다.
- 그 밖에는 evalcore.py 와 같음: IoU 0.5 greedy, COCO 101점 AP, 동의어 프롬프트 클래스별 NMS, P/R 은 op_conf 기준.

예)
  python ego_run.py --model yoloe --weights yoloe-26s-seg.pt --prompt-set name --device cpu
  python ego_run.py --model yoloworld --weights yolov8s-worldv2.pt --prompt-set name --device cpu
결과: runs_ego/<model>_<weights>_<prompt-set>/ (metrics.json, per_class.csv, per_image.json, preds/)
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from evalcore import IOU_THR, ap_101, class_nms, iou_matrix, match_image, to_xyxy

HERE = Path(__file__).resolve().parent
CATS = [(611, "wallet"), (287, "house/car key"), (353, "mobile phone"), (271, "headphones"),
        (243, "glasses"), (615, "watch"), (455, "remote control"), (408, "phone charger")]
NAMES = [n for _, n in CATS]
MAIN = ["wallet", "house/car key", "mobile phone", "headphones"]  # 우리 대상(에어팟은 headphones 로 대신)
CIDX = {cid: i for i, (cid, _) in enumerate(CATS)}


def load_data(data):
    d = json.loads((data / "annotations_val_subset.json").read_text(encoding="utf-8"))
    recs = {}
    for im in d["images"]:
        stem = Path(im["url"]).stem
        recs[stem] = {"file": str(data / "images" / im["url"]), "w": im["width"], "h": im["height"],
                      "group": im["group_id"], "gt": [], "pos": set(),
                      "neg": {CIDX[c] for c in im.get("neg_category_ids", []) if c in CIDX}, "_id": im["id"]}
    by_id = {r["_id"]: r for r in recs.values()}
    for a in d["annotations"]:
        c = a.get("category_id")
        if c not in CIDX or a["image_id"] not in by_id:
            continue
        r = by_id[a["image_id"]]
        x, y, w, h = a["bbox"]
        r["gt"].append([CIDX[c], (x + w / 2) / r["w"], (y + h / 2) / r["h"], w / r["w"], h / r["h"]])
        r["pos"].add(CIDX[c])
    for r in recs.values():
        r["gt"] = np.array(r["gt"], float).reshape(-1, 5)
    return recs


def score_image(r, preds, op_conf):
    """한 이미지 채점 결과 (부트스트랩에서 다시 합칠 수 있게 원시값으로)."""
    valid = r["pos"] | r["neg"]
    preds = class_nms(preds) if len(preds) else preds
    if len(preds):
        preds = preds[np.isin(preds[:, 0], list(valid))] if valid else preds[:0]
    status, gconf = match_image(r["gt"], preds, np.zeros((0, 5)))
    op = preds[:, 5] >= op_conf if len(preds) else np.zeros(0, bool)
    fp_op = (status == 0) & op
    wrong = 0
    if len(r["gt"]) and len(preds):
        ious = iou_matrix(to_xyxy(preds[:, 1:5]), to_xyxy(r["gt"][:, 1:5]))
        for i in np.where(fp_op)[0]:
            if ((ious[i] >= IOU_THR) & (r["gt"][:, 0] != preds[i, 0])).any():
                wrong += 1
    absent_fp = int(sum(1 for i in np.where(fp_op)[0] if int(preds[i, 0]) in r["neg"]))
    return {"group": r["group"],
            "cls": preds[:, 0].astype(int).tolist() if len(preds) else [],
            "conf": preds[:, 5].tolist() if len(preds) else [],
            "tp": status.tolist(),
            "gt_cls": r["gt"][:, 0].astype(int).tolist(),
            "gt_conf": [None if np.isnan(v) else float(v) for v in gconf],
            "fp_op": int(fp_op.sum()), "wrong_op": wrong, "absent_fp_op": absent_fp}


def aggregate(per_image, stems, op_conf):
    cls = np.array([c for s in stems for c in per_image[s]["cls"]], int)
    conf = np.array([c for s in stems for c in per_image[s]["conf"]], float)
    tp = np.array([t for s in stems for t in per_image[s]["tp"]], float)
    gcls = np.array([c for s in stems for c in per_image[s]["gt_cls"]], int)
    gconf = np.array([np.nan if c is None else c for s in stems for c in per_image[s]["gt_conf"]], float)
    hit = ~np.isnan(gconf) & (np.nan_to_num(gconf) >= op_conf)
    per = {}
    for k, name in enumerate(NAMES):
        m, n = cls == k, int((gcls == k).sum())
        op = m & (conf >= op_conf)
        tp_op = int(tp[op].sum())
        per[name] = {"n_gt": n, "AP50": ap_101(tp[m], conf[m], n),
                     "precision@op": tp_op / op.sum() if op.sum() else float("nan"),
                     "recall@op": float(hit[gcls == k].mean()) if n else float("nan"),
                     "fp@op": int(op.sum() - tp_op)}
    det = conf >= op_conf
    return {
        "images": len(stems), "sessions": len({per_image[s]["group"] for s in stems}),
        "mAP50_8cls": float(np.nanmean([per[n]["AP50"] for n in NAMES])),
        "mAP50_main4": float(np.nanmean([per[n]["AP50"] for n in MAIN])),
        "precision@op": float(tp[det].sum() / det.sum()) if det.sum() else float("nan"),
        "recall@op": float(hit.mean()) if len(hit) else float("nan"),
        "recall@op_main4": float(hit[np.isin(gcls, [NAMES.index(n) for n in MAIN])].mean()),
        "fp@op_per_image": sum(per_image[s]["fp_op"] for s in stems) / max(len(stems), 1),
        "fp@op_on_confirmed_absent": sum(per_image[s]["absent_fp_op"] for s in stems),
        "wrong_class@op": sum(per_image[s]["wrong_op"] for s in stems),
        "per_class": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yoloe", "yoloworld", "gdino"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(HERE.parents[1] / "data" / "egoobjects_val"))
    ap.add_argument("--prompt-set", default="name")
    ap.add_argument("--prompts", default=str(HERE / "prompts_ego.json"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--op-conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--speed-images", type=int, default=100, help="속도 측정용 무작위 이미지 수 (0이면 생략)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from speed import load_model, measure

    data = Path(args.data)
    recs = load_data(data)
    stems = sorted(recs)
    cfg = json.loads(Path(args.prompts).read_text(encoding="utf-8"))[args.prompt_set]
    texts, text2cls = [], []
    for c, name in enumerate(NAMES):
        for t in cfg[name]:
            texts.append(t)
            text2cls.append(c)
    model = load_model(args.model, args.weights, texts, args.device)

    run = args.out or f"{args.model}_{Path(args.weights).stem}_{args.prompt_set}"
    out = HERE / "runs_ego" / run
    (out / "preds").mkdir(parents=True, exist_ok=True)
    kw = dict(conf=0.001, imgsz=args.imgsz, verbose=False)
    if args.device is not None:
        kw["device"] = args.device

    per_image = {}
    for i, s in enumerate(stems):
        r = model.predict(recs[s]["file"], **kw)[0]
        rows = []
        if r.boxes is not None and len(r.boxes):
            for (cx, cy, w, h), k, cf in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                rows.append([text2cls[int(k)], cx, cy, w, h, cf])
        (out / "preds" / f"{s}.txt").write_text("\n".join(" ".join(f"{v:.6f}" for v in row) for row in rows))
        per_image[s] = score_image(recs[s], np.array(rows, float).reshape(-1, 6), args.op_conf)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(stems)} images", flush=True)

    if args.speed_images > 0:
        sample = random.Random(0).sample(stems, min(args.speed_images, len(stems)))
        print(f"속도 측정 중 (워밍업 후 {len(sample)}장 x {args.repeats}회)...", flush=True)
        speed = measure(model, [recs[s]["file"] for s in sample], args.imgsz, args.device, repeats=args.repeats)
    else:
        speed = {"ms_per_image_mean": float("nan"), "ms_per_image_std": float("nan"), "fps": float("nan")}

    m = aggregate(per_image, stems, args.op_conf)
    m = {"model": args.model, "weights": args.weights, "prompt_set": args.prompt_set, "prompts": cfg,
         "op_conf": args.op_conf, "scoring": "LVIS federated (pos + neg_category_ids)", "speed": speed, **m}
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "per_image.json").write_text(json.dumps(per_image), encoding="utf-8")
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_gt", "AP50", "precision@op", "recall@op", "fp@op"])
        for k, v in m["per_class"].items():
            w.writerow([k, v["n_gt"], *[round(v[x], 4) if v[x] == v[x] else "" for x in
                                        ["AP50", "precision@op", "recall@op"]], v["fp@op"]])

    sp = speed
    print(f"\n[{run}] images={m['images']} sessions={m['sessions']}  "
          f"speed={sp['ms_per_image_mean']:.1f}±{sp['ms_per_image_std']:.1f} ms/img")
    print(f"  mAP50 (8 cls)         : {m['mAP50_8cls']:.3f}")
    print(f"  mAP50 (main 4)        : {m['mAP50_main4']:.3f}")
    print(f"  precision / recall    : {m['precision@op']:.3f} / {m['recall@op']:.3f}  (recall main4 {m['recall@op_main4']:.3f})")
    print(f"  FP/img {m['fp@op_per_image']:.2f}  FP on confirmed-absent {m['fp@op_on_confirmed_absent']}  wrong-class {m['wrong_class@op']}")
    for k, v in m["per_class"].items():
        print(f"    {k:15s} n={v['n_gt']:4d} AP50={v['AP50']:.3f} P={v['precision@op']:.3f} R={v['recall@op']:.3f}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
