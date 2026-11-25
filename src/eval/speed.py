"""반복 추론으로 평균 처리 시간과 FPS를 잰다. (두 모델 공통 측정 규칙)

규칙
- batch 1, 같은 imgsz, 같은 device
- 워밍업 WARMUP 회는 측정에서 제외
- 평가 이미지 전체를 REPEATS 회 반복해서 이미지당 시간(ms)을 모두 기록
- 보고: 평균 ± 표준편차, 중앙값, p95, FPS(=1000/평균)
- end-to-end(전처리+추론+후처리) 벽시계 시간 + Ultralytics 내부 단계별 시간

단독 실행:
  python speed.py --model yoloe --weights yoloe-26s-seg.pt --prompt-set name --device cpu
  python speed.py --model yoloworld --weights yolov8s-worldv2.pt --prompt-set name --device cpu
"""
import time

import numpy as np

WARMUP = 10
REPEATS = 5


def measure(model, image_paths, imgsz=640, device=None, warmup=WARMUP, repeats=REPEATS):
    kw = dict(conf=0.001, imgsz=imgsz, verbose=False)
    if device is not None:
        kw["device"] = device
    paths = [str(p) for p in image_paths]
    for i in range(warmup):
        model.predict(paths[i % len(paths)], **kw)
    wall, pre, inf, post = [], [], [], []
    for _ in range(repeats):
        for p in paths:
            t0 = time.perf_counter()
            r = model.predict(p, **kw)[0]
            wall.append((time.perf_counter() - t0) * 1000)
            sp = getattr(r, "speed", None) or {}
            pre.append(sp.get("preprocess", np.nan))
            inf.append(sp.get("inference", np.nan))
            post.append(sp.get("postprocess", np.nan))
    w = np.array(wall)
    stat = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    return {
        "protocol": f"batch=1, imgsz={imgsz}, warmup={warmup}, repeats={repeats} x {len(paths)} images",
        "device": device or "auto",
        "n_measurements": int(len(w)),
        "ms_per_image_mean": float(w.mean()),
        "ms_per_image_std": float(w.std()),
        "ms_per_image_median": float(np.median(w)),
        "ms_per_image_p95": float(np.percentile(w, 95)),
        "fps": float(1000 / w.mean()),
        "ultralytics_ms": {"preprocess": stat(np.array(pre)), "inference": stat(np.array(inf)),
                           "postprocess": stat(np.array(post))},
    }


def load_model(kind, weights, texts, device=None):
    if kind == "gdino":  # weights = HF 모델 id (예: IDEA-Research/grounding-dino-base)
        from gdino import GroundingDINO
        return GroundingDINO(weights, texts, device=device)
    if kind == "yoloe":
        from ultralytics import YOLOE
        model = YOLOE(weights)
        try:
            model.set_classes(texts)
        except TypeError:  # 구버전 ultralytics 호환
            model.set_classes(texts, model.get_text_pe(texts))
    else:
        from ultralytics import YOLOWorld
        model = YOLOWorld(weights)
        model.set_classes(texts)
    return model


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from evalcore import NAMES

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yoloe", "yoloworld", "gdino"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(here.parents[1] / "data" / "egograsp_eval"))
    ap.add_argument("--prompt-set", default="name")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args()
    cfg = json.loads((here / "prompts.json").read_text(encoding="utf-8"))[a.prompt_set]
    texts = [t for n in NAMES for t in cfg[n]]
    imgs = sorted(Path(a.data).glob("*/images/*.jpg"))
    res = measure(load_model(a.model, a.weights, texts, a.device), imgs, a.imgsz, a.device, repeats=a.repeats)
    out = here / "runs" / f"speed_{a.model}_{Path(a.weights).stem}_{a.device or 'auto'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
