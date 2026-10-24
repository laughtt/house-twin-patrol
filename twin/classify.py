"""What is in a crop, offline, with OpenCV's dnn module: an ImageNet classifier (OpenCV model zoo) or open-vocabulary
CLIP (image half in cv.dnn; the words are embedded once by `python -m twin.classify` into service/models/clip_labels.npz).
Both answer in the home vocabulary of config.toml [landmarks]: [(label, probability), ...] best first."""
import json
import sys
from functools import lru_cache

import cv2 as cv
import numpy as np

from twin import CFG, ROOT

MODELS = ROOT / "service/models"
LM = CFG["landmarks"]
# furniture-sized words only: with "cup" or "clothes" on the list, CLIP names the small thing in a sink or on a shelf
# (a bathtub came out "bowl" at 88 %). Non-objects (blur, reflection, poster, person) come from [curate] clip_negatives.
VOCAB = LM["fixed"] + LM["rare"] + LM["often"]
# ImageNet's words → ours (only the classes a home has; the rest add nothing)
IMAGENET = {"refrigerator": "fridge", "microwave oven": "microwave", "stove": "stove", "Dutch oven": "stove", "wok": "stove",
            "frying pan": "stove", "rotisserie": "oven", "dishwasher": "dishwasher", "washing machine": "washer", "sink": "sink",
            "bathtub": "bathtub", "tub": "bathtub", "shower curtain": "shower", "toilet seat": "toilet", "couch": "sofa",
            "studio couch": "sofa", "four-poster bed": "bed", "quilt": "bed", "pillow": "bed", "infant bed": "bed", "cradle": "bed",
            "television": "tv", "monitor": "tv", "home theater": "tv", "entertainment center": "tv", "CRT screen": "tv",
            "dining table": "dining table", "desk": "desk", "desktop computer": "desk", "laptop computer": "desk",
            "computer keyboard": "desk", "wardrobe": "wardrobe", "bookcase": "bookshelf", "library": "bookshelf",
            "plate rack": "shelf", "china cabinet": "cabinet", "chiffonier": "dresser", "filing cabinet": "cabinet",
            "medicine chest": "cabinet", "chest": "cabinet", "folding chair": "chair", "rocking chair": "chair",
            "barber chair": "office chair", "throne": "armchair", "park bench": "bench", "pot": "potted plant", "vase": "vase",
            "window screen": "window", "window shade": "window", "sliding door": "door", "shoji": "door", "table lamp": "lamp",
            "lampshade": "lamp", "radiator": "radiator", "space heater": "radiator", "fire screen sheet": "fireplace",
            "grand piano": "piano", "upright piano": "piano", "bucket": "bin", "barrel": "bin", "toaster": "kitchen counter",
            "coffeemaker": "kitchen counter", "espresso machine": "kitchen counter", "Crock Pot": "kitchen counter"}


def _blob(img, mean, std):
    b = cv.dnn.blobFromImage(img, 1 / 255.0, (224, 224), swapRB=True, crop=True)
    return ((b - np.reshape(mean, (1, 3, 1, 1))) / np.reshape(std, (1, 3, 1, 1))).astype(np.float32)


@lru_cache(maxsize=None)
def _net(name):
    return cv.dnn.readNet(str(MODELS / name))


def imagenet(img, model="image_classification_ppresnet50_2022jan.onnx", top=10):
    """ImageNet classifier (OpenCV model zoo) → our labels, probabilities of the classes that map to each summed."""
    labels = json.loads((MODELS / "imagenet-simple-labels.json").read_text())
    net = _net(model)
    net.setInput(_blob(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    z = net.forward().ravel()
    p = np.exp(z - z.max())
    p /= p.sum()
    out = {}
    for i in np.argsort(-p)[:top]:
        if labels[i] in IMAGENET:
            out[IMAGENET[labels[i]]] = out.get(IMAGENET[labels[i]], 0) + float(p[i])
    return sorted(out.items(), key=lambda kv: -kv[1])


def _prompts():
    words = [(f"a photo of a {w} in a home", w) for w in VOCAB]
    return words + list(CFG["curate"]["clip_negatives"].items())


@lru_cache(maxsize=None)
def _clip_words():
    z = np.load(MODELS / "clip_labels.npz", allow_pickle=False)
    if list(z["prompts"]) != [p for p, _ in _prompts()]:
        raise RuntimeError("the vocabulary changed: run `python -m twin.classify` to embed the new words")
    return z["emb"], [str(lab) for lab in z["labels"]]


def clip(img, temperature=100.0):
    """Open-vocabulary: the crop's CLIP embedding (cv.dnn) against every word of the vocabulary and the negatives."""
    emb, labels = _clip_words()
    net = _net("clip_vision.onnx")
    net.setInput(_blob(img, [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]), "pixel_values")
    v = net.forward().ravel()
    s = temperature * emb @ (v / np.linalg.norm(v))
    p = np.exp(s - s.max())
    p /= p.sum()
    out = {}
    for lab, q in zip(labels, p):
        out[lab] = out.get(lab, 0) + float(q)
    return sorted(out.items(), key=lambda kv: -kv[1])


def embed_words():
    """Embed the vocabulary's prompts with CLIP's text half (onnxruntime + tokenizers, once per vocabulary change)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODELS / "clip_tokenizer.json"))
    tok.enable_padding(length=77, pad_id=49407)
    tok.enable_truncation(77)
    prompts = _prompts()
    enc = tok.encode_batch([p for p, _ in prompts])
    sess = ort.InferenceSession(str(MODELS / "clip_text.onnx"))
    feed = {"input_ids": np.array([e.ids for e in enc], np.int64)}
    if "attention_mask" in [i.name for i in sess.get_inputs()]:
        feed["attention_mask"] = np.array([e.attention_mask for e in enc], np.int64)
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
    emb = out["text_embeds"]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(MODELS / "clip_labels.npz", emb=emb.astype(np.float32), labels=np.array([lab for _, lab in prompts]),
             prompts=np.array([p for p, _ in prompts]))
    return len(prompts)


if __name__ == "__main__":
    print(f"embedded {embed_words()} prompts → {MODELS / 'clip_labels.npz'}")
    if len(sys.argv) > 1:
        img = cv.imread(sys.argv[1])
        print("imagenet:", imagenet(img)[:5])
        print("clip:", [(lab, round(p, 3)) for lab, p in clip(img)[:5]])
