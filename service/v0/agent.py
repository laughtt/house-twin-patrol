"""v0 agent: Claude (Bedrock) picks size references and questions from the frames; plain maths turns
model metres into real metres.

python -m v0.agent <session_dir> <room>
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from v0.scan import Scan, read_frames

MODEL = os.environ.get("V0_MODEL", "global.anthropic.claude-opus-5-5")
DEPTH_SIGMA = 0.25  # the small depth model's own metres count as a weak reference
PRO_SIGMA = 0.10  # Depth Pro's metres: stronger, but it disagreed with the small model by 23 % on a real photo
LENS_SIGMA = 0.03  # lens + phone swing, left over after every reference
PRIORS = {  # height of the object's top above the floor in US homes: typical m, spread
    "kitchen counter / base cabinet": (0.91, 0.03), "stove / range cooktop": (0.91, 0.03),
    "dining table": (0.75, 0.03), "desk": (0.74, 0.04), "coffee table": (0.45, 0.12),
    "bathroom vanity": (0.84, 0.06), "toilet tank top": (0.76, 0.07), "bathtub rim": (0.51, 0.08),
    "kitchen / dining chair backrest": (0.88, 0.08), "chair seat": (0.46, 0.06), "bar stool seat": (0.76, 0.06),
    "sofa backrest": (0.85, 0.12), "sofa seat": (0.44, 0.10), "bed mattress top": (0.60, 0.15),
    "nightstand": (0.62, 0.10), "dresser": (0.80, 0.15), "washer / dryer (front load)": (0.97, 0.04),
}
SCHEMA = {"type": "object", "required": ["room_type", "objects", "extra_refs", "questions", "eyeball_m", "warnings"],
          "properties": {
              "room_type": {"type": "string"},
              "objects": {"type": "array", "items": {"type": "object",
                          "required": ["id", "keep", "reference", "typical_height_m", "spread_pct", "why"],
                          "properties": {"id": {"type": "string"}, "keep": {"type": "boolean"},
                                         "reference": {"type": "string"}, "typical_height_m": {"type": "number"},
                                         "spread_pct": {"type": "number"}, "why": {"type": "string"}}}},
              "extra_refs": {"type": "array", "items": {"type": "object",
                             "required": ["image", "label", "box", "typical_height_m", "spread_pct", "why"],
                             "properties": {"image": {"type": "integer"}, "label": {"type": "string"},
                                            "box": {"type": "array", "items": {"type": "number"}},
                                            "typical_height_m": {"type": "number"},
                                            "spread_pct": {"type": "number"}, "why": {"type": "string"}}}},
              "questions": {"type": "array", "items": {"type": "object", "required": ["ref", "text"],
                            "properties": {"ref": {"type": "string"}, "text": {"type": "string"}}}},
              "eyeball_m": {"type": "object", "required": ["width", "length"],
                            "properties": {"width": {"type": "number"}, "length": {"type": "number"}}},
              "warnings": {"type": "array", "items": {"type": "string"}}}}
PROMPT = """These images come from one slow full turn in the middle of a room in a home (phone at chest height,
aimed a little down). Boxes labelled objN were found by an object detector: {objects}

We are measuring the room WITHOUT markers. We can measure how high above the floor the TOP of an object is, if it
stands on the floor and its top is in view. We turn our measurements into real metres with objects whose height is
standard. Typical heights of the top (US homes), as [metres, spread]: {priors}

Call the report tool with:
1. objects: one entry per objN. keep=false if it is a wrong detection, does not stand on the floor, or its top is not
   in view. Otherwise pick the closest reference from the table (or name your own), its typical top height and a spread
   in % (how standard that height is: 3 for counters and dining tables, 10-15 for sofas and beds).
2. extra_refs: up to 4 more references the detector could not label (kitchen counter, desk, vanity, bed, stove ...).
   Give the image number and a tight box around the object, [x0, y0, x1, y1] as fractions of that image's width and
   height. Only objects standing on the floor with the top in view. They will be called ref1, ref2 ... in your order.
3. questions: at most 2, each asking the human to measure the height of the top of ONE reference in cm
   (ref = its objN or refN id). Pick the references that would help most.
4. eyeball_m: your own estimate of the room's width and length in metres, from the images alone.
5. warnings: anything that makes this unreliable (open plan, mirrors, odd shape, few references, blur)."""


def ask_claude(images, objects):
    import boto3
    content = []
    for i, img in enumerate(images):
        content += [{"text": f"Image {i}"},
                    {"image": {"format": "jpeg", "source": {"bytes": cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()}}}]
    content.append({"text": PROMPT.format(objects=json.dumps(objects), priors=json.dumps(PRIORS))})
    r = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")).converse(
        modelId=MODEL, messages=[{"role": "user", "content": content}], inferenceConfig={"maxTokens": 4000},
        toolConfig={"tools": [{"toolSpec": {"name": "report", "description": "Report the size references.",
                                            "inputSchema": {"json": SCHEMA}}}]})
    return next(c["toolUse"]["input"] for c in r["output"]["message"]["content"] if "toolUse" in c)


def agent_images(scan, objects, n=6):
    # the best view of each object first, then frames spread round the circle
    ks = list(dict.fromkeys(o["k"] for o in objects))[:n]
    for k in np.linspace(0, len(scan.frames) - 1, n * 2).astype(int):
        if len(ks) < n and all(abs(k - j) > len(scan.frames) / (n * 2) for j in ks):
            ks.append(int(k))
    images = []
    for k in ks:
        img = scan.frames[k].copy()
        for o in objects:
            if o["k"] == k:
                x0, y0, x1, y1 = map(int, o["box"])
                cv.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
                cv.putText(img, f'{o["id"]} {o["cls"]}', (x0 + 4, y0 + 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        images.append(img)
    return images, ks


def fuse(evidence):
    # weighted mean in log space; drop references that disagree with the rest by more than 2.5 sigma
    use = list(evidence)
    while True:
        w = np.array([1 / e["sigma"] ** 2 for e in use])
        lg = np.array([math.log(e["scale"]) for e in use])
        mu, sd = float((w * lg).sum() / w.sum()), float(w.sum() ** -0.5)
        keep = [e for e, x in zip(use, lg) if abs(x - mu) <= 2.5 * math.hypot(e["sigma"], sd)]
        if len(keep) == len(use):
            return math.exp(mu), sd, use
        use = keep


def estimate(r):
    ev = [{"id": "depth", "label": "depth model's own metres", "scale": 1.0, "sigma": DEPTH_SIGMA}]
    if r["model"].get("pro"):
        ev.append({"id": "depthpro", "label": "Depth Pro's own metres", "scale": r["model"]["pro"]["scale"],
                   # one readable frame can't show how much it wobbles, so it only counts as much as the small model
                   "sigma": max(PRO_SIGMA, r["model"]["pro"]["spread"]) if r["model"]["pro"].get("used", 0) >= 2 else DEPTH_SIGMA})
    for f in r["refs"]:
        cm = r["answers"].get(f["id"])
        typical, spread = (cm / 100, 0.015) if cm else (f["typical_m"], f["spread"])
        ev.append({"id": f["id"], "label": f["label"] + (" (your measure)" if cm else ""),
                   "scale": typical / f["measured_m"],
                   "sigma": math.hypot(spread, 0.04 + 0.08 / math.sqrt(f["views"]))})
    scale, sd, used = fuse(ev)
    sd = math.hypot(sd, LENS_SIGMA)
    w, l = r["model"]["size_m"]
    open_q = [q for q in r["questions"] if q["ref"] not in r["answers"]]
    return {"scale": scale, "sigma_pct": round(100 * sd, 1), "size_m": [round(scale * w, 2), round(scale * l, 2)],
            "evidence": [{**e, "used": e in used} for e in ev],
            "status": "needs_answer" if sd > 0.05 and open_q else "done"}


def run(folder, room):
    out = folder / "v0"
    out.mkdir(exist_ok=True)
    path = out / f"{room}.json"
    path.write_text(json.dumps({"status": "running", "room": room, "t": time.time()}))
    try:
        video = next(p for p in folder.iterdir() if p.stem == room and p.suffix in (".mp4", ".webm", ".mov"))
        t0 = time.time()
        s = Scan(read_frames(video))
        objs = s.objects()
        images, ks = agent_images(s, objs)
        for i, img in enumerate(images):
            cv.imwrite(str(out / f"{room}-img{i}.jpg"), img)
        s.top_view(out / f"{room}-top.png", objs)
        t1 = time.time()
        ai = ask_claude(images, [{"id": o["id"], "cls": o["cls"], "image": ks.index(o["k"])} for o in objs if o["k"] in ks])
        byid = {o["id"]: o for o in objs}
        refs = [{"id": a["id"], "label": a["reference"], "measured_m": byid[a["id"]]["top_model_m"],
                 "typical_m": a["typical_height_m"], "spread": a["spread_pct"] / 100,
                 "views": byid[a["id"]]["views"], "why": a["why"]}
                for a in ai["objects"] if a["keep"] and a["id"] in byid]
        for n, e in enumerate(ai["extra_refs"]):
            k = ks[e["image"]] if 0 <= e["image"] < len(ks) else None
            h, w = s.frames[0].shape[:2]
            m = k is not None and len(e["box"]) == 4 and s.measure(k, [e["box"][0] * w, e["box"][1] * h, e["box"][2] * w, e["box"][3] * h])
            if m:
                refs.append({"id": f"ref{n + 1}", "label": e["label"], "measured_m": m[0], "typical_m": e["typical_height_m"],
                             "spread": e["spread_pct"] / 100, "views": 1, "why": e["why"], "image": e["image"], "box": e["box"]})
        result = {"room": room, "video": video.name, "room_type": ai["room_type"], "refs": refs,
                  "questions": ai["questions"][:2], "answers": {}, "eyeball_m": ai["eyeball_m"], "warnings": ai["warnings"],
                  "images": len(images), "model": {"size_m": [round(v, 2) for v in s.size], "hfov_deg": round(s.hfov, 1),
                  "cam_height_m": round(s.cam_h, 2), "spin_deg": round(s.spin_deg), "frames": len(s.frames), "pro": s.pro,
                  "size_pro_m": [round(v * s.pro["scale"], 2) for v in s.size] if s.pro else None},
                  "seconds": {"vision": round(t1 - t0, 1), "agent": round(time.time() - t1, 1)}, "llm": MODEL}
        if s.pro and abs(s.pro["hfov_deg"] - s.hfov) > 0.05 * s.hfov:
            result["warnings"].append(f"Lens check disagrees: the turn says {s.hfov:.0f}°, Depth Pro says {s.pro['hfov_deg']:.0f}°.")
        result.update(estimate(result))
    except Exception as e:
        result = {"status": "error", "room": room, "error": f"{type(e).__name__}: {e}"}
    path.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), sys.argv[2]), indent=1))
