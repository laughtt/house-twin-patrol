"""Flight recorder for Pollen's duck-sim: reads the body server (the same `read` robotd uses) at 25 Hz → body.npz
(time, trunk xyz, quat, the 15 wire joints) for the film, and the latest true pose → truth.json for agent/walk.py.

python -m agent.duck_recorder --out out/walk/<house> [--port 7801]
"""
import argparse
import json
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(prog="agent.duck_recorder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=7801)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = socket.create_connection(("127.0.0.1", a.port))
    f = s.makefile("rw")
    f.write(json.dumps({"op": "hello", "protocol": 1, "joints": 15}) + "\n")
    f.flush()
    f.readline()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit())  # pkill: still save
    log = []
    try:
        while True:
            tick = time.time()
            f.write('{"op":"read"}\n')
            f.flush()
            r = json.loads(f.readline())
            w, x, y, z = r["imu"]["quat"]
            log.append([tick, *r["trunk"], w, x, y, z, *r["positions"]])
            (out / "truth.tmp").write_text(json.dumps({"t": tick, "xyz": r["trunk"],
                                                       "yaw": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))}))
            os.replace(out / "truth.tmp", out / "truth.json")
            if len(log) % 250 == 0:
                np.save(out / "body.npy", np.array(log))
            time.sleep(max(0.0, 0.04 - (time.time() - tick)))
    except (KeyboardInterrupt, ValueError):  # stopped, or the simulator went away
        pass
    finally:
        np.save(out / "body.npy", np.array(log))


if __name__ == "__main__":
    main()
