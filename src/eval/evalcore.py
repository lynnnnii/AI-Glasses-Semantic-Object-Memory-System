"""모델과 무관한 공통 채점 코드 (numpy만 사용).

두 모델의 예측을 똑같은 규칙으로 채점하기 위해 Ultralytics의 val() 대신 이 코드를 쓴다.
- 박스 형식: 정규화 xywh (YOLO txt와 동일)
- 예측 txt: "cls cx cy w h conf"
- 매칭: 같은 클래스, IoU >= 0.5, conf 높은 순으로 greedy
- ignore 박스와 IoU >= 0.5 인 미매칭 예측은 TP도 FP도 아님 (버림)
"""
import csv
import json
from pathlib import Path

import numpy as np

NAMES = ["bottle", "bucket", "faucet", "mug", "pliers", "smartphone"]
IOU_THR = 0.5


def read_boxes(path, ncol):
    p = Path(path)
    if not p.exists():
        return np.zeros((0, ncol))
    rows = [list(map(float, ln.split())) for ln in p.read_text().split("\n") if ln.strip()]
    return np.array(rows, dtype=float).reshape(-1, ncol) if rows else np.zeros((0, ncol))


def to_xyxy(xywh):
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def class_nms(preds, iou_thr=0.6):
    """같은 GT 클래스로 매핑된 동의어 프롬프트들의 중복 박스 제거. preds: (N,6) cls,cx,cy,w,h,conf"""
    if len(preds) == 0:
        return preds
    keep = []
    for c in np.unique(preds[:, 0]):
        p = preds[preds[:, 0] == c]
        p = p[np.argsort(-p[:, 5])]
        boxes = to_xyxy(p[:, 1:5])
        alive = np.ones(len(p), bool)
        for i in range(len(p)):
            if not alive[i]:
                continue
            keep.append(p[i])
            alive &= ~(iou_matrix(boxes[i:i + 1], boxes)[0] >= iou_thr)
            alive[i] = False
    return np.array(keep)


def match_image(gt, preds, ignore):
    """한 이미지 채점. return: 예측별 상태(1=TP, 0=FP, -1=ignored), GT별 매칭된 예측 conf(없으면 nan)"""
    status = np.zeros(len(preds), int)
    gt_conf = np.full(len(gt), np.nan)
    if len(preds) == 0:
        return status, gt_conf
    order = np.argsort(-preds[:, 5])
    ious = iou_matrix(to_xyxy(preds[:, 1:5]), to_xyxy(gt[:, 1:5])) if len(gt) else np.zeros((len(preds), 0))
    ign_ious = iou_matrix(to_xyxy(preds[:, 1:5]), to_xyxy(ignore[:, 1:5])) if len(ignore) else np.zeros((len(preds), 0))
    used = np.zeros(len(gt), bool)
    for i in order:
        best, best_j = IOU_THR, -1
        for j in range(len(gt)):
            if used[j] or gt[j, 0] != preds[i, 0]:
                continue
            if ious[i, j] >= best:
                best, best_j = ious[i, j], j
        if best_j >= 0:
            used[best_j] = True
            status[i] = 1
            gt_conf[best_j] = preds[i, 5]
        elif ign_ious.shape[1] and (ign_ious[i] >= IOU_THR).any():
            status[i] = -1
    return status, gt_conf


def ap_101(tp_flags, confs, n_gt):
    """COCO 방식 101-point interpolated AP"""
    if n_gt == 0:
        return float("nan")
    if len(confs) == 0:
        return 0.0
    o = np.argsort(-confs)
    tp = np.cumsum(tp_flags[o])
    fp = np.cumsum(1 - tp_flags[o])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, t, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return ap / 101


def evaluate(data_root, pred_dir, splits, op_conf=0.25, objects_csv=None):
    data_root, pred_dir = Path(data_root), Path(pred_dir)
    tag_of = {}
    if objects_csv is None:
        objects_csv = data_root / "objects.csv"
    if Path(objects_csv).exists():
        with open(objects_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["image"].rsplit(".", 1)[0], NAMES.index(r["class"]),
                       round(float(r["cx"]), 4), round(float(r["cy"]), 4))
                tag_of[key] = r["hand_overlap"]

    per_cls = {c: {"tp": [], "conf": [], "n_gt": 0} for c in range(len(NAMES))}
    gt_records = []  # (class, tag, matched_conf)
    wrong_class = 0
    neg_images, neg_fp = 0, 0
    n_images, fp_total = 0, 0
    for split in splits:
        for img in sorted((data_root / split / "images").glob("*.jpg")):
            stem = img.stem
            n_images += 1
            gt = read_boxes(data_root / split / "labels" / f"{stem}.txt", 5)
            ign = read_boxes(data_root / split / "ignore" / f"{stem}.txt", 5)
            preds = class_nms(read_boxes(pred_dir / split / f"{stem}.txt", 6))
            status, gt_conf = match_image(gt, preds, ign)
            for c in range(len(NAMES)):
                m = preds[:, 0] == c if len(preds) else np.zeros(0, bool)
                valid = status[m] >= 0
                per_cls[c]["tp"] += list(status[m][valid])
                per_cls[c]["conf"] += list(preds[m][valid, 5])
                per_cls[c]["n_gt"] += int((gt[:, 0] == c).sum()) if len(gt) else 0
            for j, g in enumerate(gt):
                tag = tag_of.get((stem, int(g[0]), round(g[1], 4), round(g[2], 4)), "unknown")
                gt_records.append((int(g[0]), tag, gt_conf[j]))
            op = (preds[:, 5] >= op_conf) if len(preds) else np.zeros(0, bool)
            fps_here = int(((status == 0) & op).sum())
            fp_total += fps_here
            if len(gt) == 0:
                neg_images += 1
                neg_fp += fps_here
            # 다른 클래스로 잘못 부른 경우: FP 중 다른 클래스 GT와 IoU>=0.5
            if len(gt) and len(preds):
                ious = iou_matrix(to_xyxy(preds[:, 1:5]), to_xyxy(gt[:, 1:5]))
                for i in np.where((status == 0) & op)[0]:
                    if ((ious[i] >= IOU_THR) & (gt[:, 0] != preds[i, 0])).any():
                        wrong_class += 1

    classes = {}
    for c, d in per_cls.items():
        tp, conf = np.array(d["tp"], float), np.array(d["conf"], float)
        op = conf >= op_conf
        tp_op = int(tp[op].sum())
        fp_op = int(op.sum() - tp_op)
        n = d["n_gt"]
        classes[NAMES[c]] = {
            "n_gt": n,
            "AP50": ap_101(tp, conf, n),
            "precision@op": tp_op / (tp_op + fp_op) if tp_op + fp_op else float("nan"),
            "recall@op": tp_op / n if n else float("nan"),
            "fp@op": fp_op,
        }

    def recall_by(tag):
        recs = [r for r in gt_records if r[1] == tag]
        if not recs:
            return {"n": 0, "recall@op": float("nan")}
        hit = sum(1 for r in recs if not np.isnan(r[2]) and r[2] >= op_conf)
        return {"n": len(recs), "recall@op": hit / len(recs)}

    tp_all = sum(int(np.array(d["tp"], float)[np.array(d["conf"], float) >= op_conf].sum()) for d in per_cls.values())
    det_all = sum(int((np.array(d["conf"], float) >= op_conf).sum()) for d in per_cls.values())
    aps = [v["AP50"] for v in classes.values() if v["n_gt"] > 0]
    aps5 = [v["AP50"] for v in classes.values() if v["n_gt"] >= 5]
    all_tp = sum(1 for r in gt_records if not np.isnan(r[2]) and r[2] >= op_conf)
    return {
        "splits": splits,
        "op_conf": op_conf,
        "images": n_images,
        "mAP50_all_classes": float(np.mean(aps)) if aps else float("nan"),
        "mAP50_classes_ge5gt": float(np.mean(aps5)) if aps5 else float("nan"),
        "precision@op_overall": tp_all / det_all if det_all else float("nan"),
        "recall@op_overall": all_tp / len(gt_records) if gt_records else float("nan"),
        "recall@op_hand_overlap": recall_by("hand"),
        "recall@op_free": recall_by("free"),
        "fp@op_total": fp_total,
        "fp@op_per_image": fp_total / max(n_images, 1),
        "fp@op_on_negative_images": {"images": neg_images, "fp": neg_fp},
        "wrong_class@op": wrong_class,
        "per_class": classes,
    }


def save_report(metrics, out_dir, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if extra:
        metrics = {**extra, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float))
    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_gt", "AP50", "precision@op", "recall@op", "fp@op"])
        for k, v in metrics["per_class"].items():
            w.writerow([k, v["n_gt"], *[round(v[x], 4) if v[x] == v[x] else "" for x in
                                        ["AP50", "precision@op", "recall@op"]], v["fp@op"]])
