"""Grounding DINO(HF transformers)를 Ultralytics predict() 와 같은 모양으로 감싼 어댑터.

run_model.py / speed.py 가 YOLO 모델과 똑같이 다룰 수 있도록
  model.predict(path, conf=..., imgsz=..., verbose=..., device=...)[0]
  -> r.boxes.xywhn / r.boxes.cls / r.boxes.conf, r.speed
를 흉내 낸다.

공정성을 위한 규칙
- 프롬프트: 모든 프롬프트를 "a . b . c ." 한 문장으로 넣어 이미지당 forward 1회
  (HF pipeline 은 라벨마다 forward 를 따로 돌려 느리고 YOLO 와 조건이 달라서 쓰지 않음)
- 점수: 쿼리별로 각 프롬프트 토큰 구간의 sigmoid 평균 (공식 Grounding DINO COCO 평가 방식),
  가장 높은 프롬프트 하나를 클래스로 (Ultralytics NMS 기본값 multi_label=False 와 동일)
- 상위 max_det=300 개만 남김 (Ultralytics 기본 max_det 과 동일), conf 이하 제거
- 입력 크기: imgsz 무시, 모델 기본 전처리(짧은 변 800, 긴 변 1333) 사용
- 시간: speed 의 preprocess/inference/postprocess 는 Ultralytics 와 같은 구간(ms)
"""
import time

import numpy as np
import torch
from PIL import Image


class _Boxes:
    def __init__(self, xywhn, cls, conf):
        self.xywhn, self.cls, self.conf = xywhn, cls, conf

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, boxes, speed):
        self.boxes, self.speed = boxes, speed


class GroundingDINO:
    def __init__(self, model_id, texts, device=None, max_det=300):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = torch.device("cpu" if device in (None, "cpu") else
                                   (f"cuda:{device}" if str(device).isdigit() else device))
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.max_det = max_det
        self.texts = list(texts)

        # 캡션과 프롬프트별 토큰 구간
        parts, spans, pos = [], [], 0
        for t in self.texts:
            t = t.lower().strip()
            spans.append((pos, pos + len(t)))
            parts.append(t)
            pos += len(t) + 3  # " . "
        self.caption = " . ".join(parts) + " ."
        enc = self.processor.tokenizer(self.caption, return_offsets_mapping=True)
        self.input_ids = enc["input_ids"]
        pmap = np.zeros((len(self.texts), len(self.input_ids)), np.float32)
        for k, (s, e) in enumerate(spans):
            for j, (a, b) in enumerate(enc["offset_mapping"]):
                if b > a and a >= s and b <= e:
                    pmap[k, j] = 1
            assert pmap[k].sum() > 0, f"프롬프트 토큰을 찾지 못함: {self.texts[k]}"
        self.pmap = torch.from_numpy(pmap / pmap.sum(1, keepdims=True))

    @torch.inference_mode()
    def predict(self, source, conf=0.25, imgsz=None, verbose=False, device=None, **_):
        t0 = time.perf_counter()
        image = Image.open(source).convert("RGB")
        inputs = self.processor(images=image, text=self.caption, return_tensors="pt").to(self.device)
        assert inputs["input_ids"][0].tolist() == self.input_ids
        t1 = time.perf_counter()
        out = self.model(**inputs)
        t2 = time.perf_counter()
        prob = out.logits[0, :, : self.pmap.shape[1]].sigmoid().cpu()  # (Q, L)
        scores = prob @ self.pmap.T  # (Q, n_texts)
        conf_q, cls_q = scores.max(1)
        boxes = out.pred_boxes[0].cpu().clamp(0, 1)  # cx cy w h, 원본 이미지 기준 정규화
        keep = conf_q >= conf
        conf_q, cls_q, boxes = conf_q[keep], cls_q[keep], boxes[keep]
        order = conf_q.argsort(descending=True)[: self.max_det]
        t3 = time.perf_counter()
        speed = {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000}
        return [_Result(_Boxes(boxes[order].numpy(), cls_q[order].numpy(), conf_q[order].numpy()), speed)]
