"""Talk to a duck (Pollen's duck-sim, or a robot) over its JSON-RPC Unix sockets: odometry and ToF streams, camera frames, move."""
import json
import os
import socket
import threading
from pathlib import Path

import cv2 as cv
import numpy as np

STATE = Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache/duck-sim"))
ROTATE = {90: cv.ROTATE_90_CLOCKWISE, 180: cv.ROTATE_180, 270: cv.ROTATE_90_COUNTERCLOCKWISE}


def _call(sock, method, rid=1):
    sock.sendall((json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": {}}) + "\n").encode())


class Duck:
    def __init__(self, name="duck-a"):
        self.paths = {k: STATE / f"{name}{s}.sock" for k, s in (("robot", ""), ("tof", "-tof"), ("frame", "-frame"))}
        self.state, self.tof = None, None
        threading.Thread(target=self._stream, args=("robot", "robot.subscribe", "robot.state", "state"), daemon=True).start()
        threading.Thread(target=self._stream, args=("tof", "tof.stream", "tof.frame", "tof"), daemon=True).start()
        self._robot = socket.socket(socket.AF_UNIX)
        self._robot.connect(str(self.paths["robot"]))

    def _stream(self, which, method, notification, attr):
        s = socket.socket(socket.AF_UNIX)
        s.connect(str(self.paths[which]))
        _call(s, method)
        for line in s.makefile("rb"):
            msg = json.loads(line)
            if msg.get("method") == notification:
                setattr(self, attr, msg["params"])

    def move(self, vx, vyaw):
        """A velocity intent; robotd zeroes it unless repeated within 500 ms."""
        self._robot.sendall((json.dumps({"jsonrpc": "2.0", "method": "robot.move",
                                         "params": {"vx": vx, "vy": 0.0, "vyaw": vyaw}}) + "\n").encode())

    def enable(self, on=True):
        """Torque on: the standing policy gets the duck up (off: it sits)."""
        self._robot.sendall((json.dumps({"jsonrpc": "2.0", "id": 2, "method": "robot.enable", "params": {"on": on}}) + "\n").encode())

    def frame(self):
        """One camera frame, turned upright (BGR), and mediad's header."""
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(3)
        s.connect(str(self.paths["frame"]))
        _call(s, "media.frame")
        buf = b""
        while b"\n" not in buf:
            buf += s.recv(1 << 16)
        head, buf = buf.split(b"\n", 1)
        h = json.loads(head)
        h = h.get("result", h)
        while len(buf) < h["bytes"]:
            buf += s.recv(1 << 20)
        img = np.frombuffer(buf[:h["bytes"]], np.uint8).reshape(h["height"], h["width"], 2)
        bgr = cv.cvtColor(img, cv.COLOR_YUV2BGR_UYVY)
        return (cv.rotate(bgr, ROTATE[h["rotate"]]) if h.get("rotate") in ROTATE else bgr), h
