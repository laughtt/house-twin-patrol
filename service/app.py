import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from v0.agent import estimate

ROOT = Path(__file__).parent
UPLOADS = Path(os.environ.get("UPLOAD_DIR", ROOT / "uploads"))
API_KEY = os.environ["API_KEY"]
MARKER_M = float(os.environ.get("MARKER_SIZE_M", "0.15"))
HFOV_DEG = 66.0  # typical phone main lens; replaced by calib.yaml once the lens check is processed
SAFE = re.compile(r"^[a-z0-9_-]{1,40}$")


def corners_m(marker_id: int):
    # the id tells the printed size: 10-29 filming/doorway 10 cm, 30-39 big far-wall 20 cm, rest 15 cm
    size = 0.10 if 10 <= marker_id < 30 else 0.20 if 30 <= marker_id < 40 else MARKER_M
    s = size / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)


detector = cv.aruco.ArucoDetector(
    cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50), cv.aruco.DetectorParameters()
)
app = FastAPI()
job = None  # one v0 measurement at a time (t4g.small has 1.8 GB)
build = None  # one LiDAR house build at a time


def check(key: str):
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(401, "bad key")


@app.get("/")
def index():
    return FileResponse(ROOT / "static/capture.html", headers={"Cache-Control": "no-cache"})


@app.post("/perceive")
async def perceive(frame: UploadFile = File(...), x_api_key: str = Header("")):
    check(x_api_key)
    t0 = time.perf_counter()
    img = cv.imdecode(np.frombuffer(await frame.read(), np.uint8), cv.IMREAD_GRAYSCALE)
    t1 = time.perf_counter()
    corners, ids, _ = detector.detectMarkers(img)
    t2 = time.perf_counter()
    h, w = img.shape
    f = w / 2 / np.tan(np.radians(HFOV_DEG) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    markers = []
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        obj = corners_m(int(i))
        _, rvec, tvec = cv.solvePnP(obj, c[0], K, None, flags=cv.SOLVEPNP_IPPE_SQUARE)
        proj, _ = cv.projectPoints(obj, rvec, tvec, K, None)
        x, _, z = tvec.ravel()
        markers.append({
            "id": int(i),
            "dist_m": round(float(np.linalg.norm(tvec)), 2),
            "bearing_deg": round(float(np.degrees(np.arctan2(x, z))), 1),
            "reproj_px": round(float(np.abs(proj.reshape(-1, 2) - c[0]).mean()), 2),
            "corners": c[0].round(1).tolist(),
        })
    t3 = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 1)
    return {
        "markers": markers, "pose": None, "obstacle": None, "decision": None, "ask_human": None,
        "cv_version": cv.__version__,
        "latency_ms": {"decode": ms(t0, t1), "aruco": ms(t1, t2), "pnp": ms(t2, t3), "total": ms(t0, t3)},
    }


@app.post("/upload")
async def upload(request: Request, session: str, room: str, ext: str = "mp4", seen: str = "",
                 x_api_key: str = Header("")):
    check(x_api_key)
    if not all(SAFE.match(v) for v in (session, room, ext)):
        raise HTTPException(400, "bad name")
    folder = UPLOADS / session
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{room}.{ext}"
    size = 0
    with open(path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    with open(folder / "manifest.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "room": room, "file": path.name, "bytes": size,
                            "seen": seen, "ua": request.headers.get("user-agent")}) + "\n")
    if room == "lidar" and ext == "zip":  # the house builds itself: no person or Claude Code in the loop
        try:
            start_build(session)
        except HTTPException:  # another build is running: Build again from /lidar
            pass
    return {"ok": True, "bytes": size}


def start_build(session):
    """python -m twin build on the uploaded scan → uploads/<session>/house/ (house.json, top.png, report.json…)."""
    global build
    if build and build.poll() is None:
        raise HTTPException(409, "busy building another house, try again in a minute")
    folder = UPLOADS / session
    (folder / "house").mkdir(exist_ok=True)
    (folder / "house" / "report.json").unlink(missing_ok=True)
    log = open(folder / "build.log", "w")
    tags = ["--tags", str(folder / "web_tags.json")] if (folder / "web_tags.json").exists() else []
    build = subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "twin", "build", str(folder / "lidar.zip"), "--out", str(folder / "house"), *tags],
                             cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
    build.session = session


@app.get("/lidar")
def lidar_page():
    return FileResponse(ROOT / "static/lidar.html", headers={"Cache-Control": "no-cache"})


@app.get("/lidar/sessions")
def lidar_sessions(x_api_key: str = Header("")):
    check(x_api_key)
    return sorted((p.parent.name for p in UPLOADS.glob("*/lidar.zip")), reverse=True)


@app.get("/lidar/status")
def lidar_status(session: str, x_api_key: str = Header("")):
    check(x_api_key)
    folder = session_dir(session)
    if not (folder / "lidar.zip").exists():
        raise HTTPException(404, "no LiDAR scan in this session")
    out = folder / "house"
    if (out / "report.json").exists():
        house = json.loads((out / "house.json").read_text())
        return {"state": "done", "report": json.loads((out / "report.json").read_text()), "landmarks": house["landmarks"],
                "candidates": json.loads((out / "candidates.json").read_text())}
    if build and build.poll() is None and build.session == session:
        return {"state": "building"}
    log = (folder / "build.log").read_text()[-1500:] if (folder / "build.log").exists() else ""
    return {"state": "failed" if log else "not built", "log": log}


@app.post("/lidar/build")
def lidar_build(session: str, x_api_key: str = Header("")):
    check(x_api_key)
    if not (session_dir(session) / "lidar.zip").exists():
        raise HTTPException(404, "no LiDAR scan in this session")
    start_build(session)
    return {"ok": True}


def scan(session):
    sys.path.insert(0, str(ROOT.parent))
    from twin.scan_io import Export
    return Export(session_dir(session) / "lidar.zip")


def upright(ex, f):
    """cv.rotate code that shows a frame the way the person saw it (gravity up), or None."""
    from twin.landmarks import _rotation
    _, P = ex.load(f)
    return _rotation(P[:3, :3].T @ [0, 0, 1])


@app.get("/lidar/frames")
def lidar_frames(session: str, x_api_key: str = Header("")):
    """Every 8th frame (2 s) with normal tracking: the photos the results page lets you tag."""
    check(x_api_key)
    ex = scan(session)
    return [{"i": f["i"], "t": round(f["t"], 1), "room": f["room"]} for f in [f for f in ex.frames if f["tracking"] == "normal"][::8]]


@app.get("/lidar/frame")
def lidar_frame(session: str, i: int, size: int = 480, x_api_key: str = Header("")):
    check(x_api_key)
    ex = scan(session)
    f = next((f for f in ex.frames if f["i"] == i), None)
    if f is None:
        raise HTTPException(404, "no such frame")
    img, rot = ex.rgb(f), upright(ex, f)
    img = img if rot is None else cv.rotate(img, rot)
    s = min(1.0, max(64, min(size, 960)) / max(img.shape[:2]))
    img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    from fastapi.responses import Response
    return Response(cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), media_type="image/jpeg")


@app.get("/lidar/tags")
def lidar_tags(session: str, x_api_key: str = Header("")):
    check(x_api_key)
    path = session_dir(session) / "web_tags.json"
    return json.loads(path.read_text()) if path.exists() else []


@app.post("/lidar/tags")
async def lidar_save_tags(session: str, request: Request, x_api_key: str = Header("")):
    """Tags made on the results page → web_tags.json (the app's format), then the house is built again.
    A tap on an upright photo (screen=true) is turned back into the saved picture's coordinates."""
    check(x_api_key)
    ex = scan(session)
    by_i = {f["i"]: f for f in ex.frames}
    back = {cv.ROTATE_90_CLOCKWISE: lambda u, v: (v, 1 - u), cv.ROTATE_90_COUNTERCLOCKWISE: lambda u, v: (1 - v, u),
            cv.ROTATE_180: lambda u, v: (1 - u, 1 - v), None: lambda u, v: (u, v)}
    tags = []
    for t in (await request.json())[:200]:
        f = by_i.get(t.get("i"))
        uv, label = t.get("uv"), str(t.get("label", "")).strip()[:40]
        if f is None or not label or not (isinstance(uv, list) and len(uv) == 2 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in uv)):
            raise HTTPException(400, "bad tag")
        u, v = back[upright(ex, f)](*uv) if t.get("screen") else uv
        tags.append({"i": f["i"], "uv": [round(u, 4), round(v, 4)], "label": label, "room": f["room"]})
    (session_dir(session) / "web_tags.json").write_text(json.dumps(tags, indent=1))
    start_build(session)
    return {"ok": True, "tags": len(tags)}


@app.get("/lidar/file")
def lidar_file(session: str, name: str, x_api_key: str = Header("")):
    check(x_api_key)
    if name not in {"top.png", "house.json", "report.json", "candidates.json", "cloud.bin", "stage_floor.png", "stage_occupancy.png", "mission.json"}:
        raise HTTPException(400, "bad name")
    return FileResponse(session_dir(session) / "house" / name)


@app.get("/lidar/capture")
def lidar_capture(session: str, x_api_key: str = Header("")):
    """Step 1 as data: every 2nd frame's camera height above the floor under it, tilt, zone and tracking; the zone taps
    and tags; the app's own score card (session.json score)."""
    check(x_api_key)
    ex = scan(session)
    out = session_dir(session) / "house"
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else None
    ground = None
    if rep and (out / "occupancy.npz").exists():
        z = np.load(out / "occupancy.npz")
        ground = (z["ground_lo"], float(z["ground_tile"]), z["ground"])
    frames = []
    for f in ex.frames[::2]:
        T = np.array(f["T"]).reshape(4, 4)
        h, pitch = None, float(np.degrees(np.arcsin(np.clip(-T[1, 2], -1, 1))))  # the camera looks along -z
        if rep:
            sf = rep["scan_frame"]
            c, s_ = np.cos(sf["theta"]), np.sin(sf["theta"])
            x, y = T[0, 3], -T[2, 3]
            x, y = c * x + s_ * y - sf["origin"][0], -s_ * x + c * y - sf["origin"][1]
            h = T[1, 3] - sf["z0"]
            if ground:
                from twin.fuse import ground_at
                h -= float(ground_at(ground, [x, y]))
            h = round(float(h), 3)
        frames.append({"t": round(f["t"], 1), "h": h, "h_app": f.get("h"), "pitch": round(pitch), "room": f["room"], "tracking": f["tracking"]})
    meta = ex.json("session.json") or {}
    return {"frames": frames, "events": [{k: e.get(k) for k in ("t", "event", "room", "label")} for e in ex.events()],
            "tags": ex.tags(), "score": meta.get("score"), "format": meta.get("format", 1),
            "low_h": __import__("twin").CFG["landmarks"]["low_h"], "mb": round((session_dir(session) / "lidar.zip").stat().st_size / 1e6)}


@app.get("/lidar/flow")
def lidar_flow():
    return FileResponse(ROOT / "static/flow.html", headers={"Cache-Control": "no-cache"})


@app.get("/lidar/mission")
def lidar_mission():
    return FileResponse(ROOT / "static/mission.html", headers={"Cache-Control": "no-cache"})


@app.get("/lidar/how")
def lidar_how():
    return FileResponse(ROOT / "static/how.html", headers={"Cache-Control": "no-cache"})


@app.get("/lidar/3d")
def lidar_3d():
    return FileResponse(ROOT / "static/house3d.html", headers={"Cache-Control": "no-cache"})


def session_dir(session: str, room: str = "x"):
    if not (SAFE.match(session) and SAFE.match(room)):
        raise HTTPException(400, "bad name")
    return UPLOADS / session


@app.get("/v0")
def v0_page():
    return FileResponse(ROOT / "static/v0.html", headers={"Cache-Control": "no-cache"})


@app.get("/v0/session")
def v0_session(session: str, x_api_key: str = Header("")):
    check(x_api_key)
    folder = session_dir(session)
    if not (folder / "manifest.jsonl").exists():
        raise HTTPException(404, "no such session")
    clips = {}
    for line in (folder / "manifest.jsonl").read_text().splitlines():
        m = json.loads(line)
        if m["room"] != "calib":
            clips[m["room"]] = m["file"]
    rooms = []
    for room, file in clips.items():
        path = folder / "v0" / f"{room}.json"
        r = json.loads(path.read_text()) if path.exists() else None
        if r and r["status"] == "running" and (job is None or job.poll() is not None):
            r = {"status": "error", "error": "the measurement stopped (out of memory?)"}
        rooms.append({"room": room, "file": file, "result": r})
    return {"session": session, "rooms": rooms}


@app.post("/v0/run")
def v0_run(session: str, room: str, x_api_key: str = Header("")):
    global job
    check(x_api_key)
    folder = session_dir(session, room)
    if job and job.poll() is None:
        raise HTTPException(409, "busy measuring another room, try again in a minute")
    (folder / "v0").mkdir(exist_ok=True)
    (folder / "v0" / f"{room}.json").write_text(json.dumps({"status": "running", "room": room, "t": time.time()}))
    job = subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "v0.agent", str(folder), room], cwd=ROOT)
    return {"ok": True}


@app.post("/v0/answer")
def v0_answer(session: str, room: str, ref: str, cm: float, x_api_key: str = Header("")):
    check(x_api_key)
    path = session_dir(session, room) / "v0" / f"{room}.json"
    r = json.loads(path.read_text())
    if ref not in {f["id"] for f in r.get("refs", [])} or not 5 <= cm <= 300:
        raise HTTPException(400, "bad answer")
    r["answers"][ref] = cm
    r.update(estimate(r))
    path.write_text(json.dumps(r, indent=1))
    return r


@app.get("/v0/file")
def v0_file(session: str, room: str, name: str, x_api_key: str = Header("")):
    check(x_api_key)
    if name not in {"top.png", *(f"img{i}.jpg" for i in range(6))}:
        raise HTTPException(400, "bad name")
    return FileResponse(session_dir(session, room) / "v0" / f"{room}-{name}")
