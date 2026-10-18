"""
Real-Time Highway Incident Detection & GIS Command Center  (backend)
====================================================================

Pipeline
--------
  video source --> YOLOv8 detect + ByteTrack --> ROI filter --> stationary-vehicle
  logic --> SocketIO events --> Leaflet dashboard (templates/index.html)

Quick start
-----------
  pip install -r requirements.txt
  python app.py --source path/to/highway.mp4        # video file
  python app.py --source rtsp://user:pass@host/live # RTSP / HTTP stream
  python app.py --source 0                          # webcam
  python app.py --demo                              # no video / no YOLO needed

  Then open http://127.0.0.1:5000

If the video file is missing, the app automatically falls back to --demo mode
(a synthetic highway scene) so the dashboard always has something to show.

Tuning for YOUR video
---------------------
  * ROI_NORM       : polygon of the active lanes, in normalised (0-1) coordinates.
                     Watch the annotated feed on the dashboard and adjust it.
  * STOP_SECONDS   : dwell time before a "Stopped Vehicle" incident (default 5 s).
  * STOP_RADIUS_FRAC: how far (fraction of frame width) a vehicle may drift and
                     still count as "stationary".
"""

import argparse
import math
import os
import random
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_WEIGHTS = os.getenv("YOLO_MODEL", "yolov8n.pt")   # auto-downloads on first run
CONFIDENCE = 0.35
COCO_VEHICLES = {2: "car", 5: "bus", 7: "truck"}         # COCO class id -> label
CLASS_NAMES = list(COCO_VEHICLES.values())

# Region of Interest: active highway lanes as a polygon in normalised (x, y).
# Default is a trapezoid typical of a forward-facing gantry/overpass camera.
ROI_NORM = [(0.05, 0.97), (0.95, 0.97), (0.64, 0.45), (0.36, 0.45)]

STOP_SECONDS = 5.0          # dwell time that triggers an incident
STOP_RADIUS_FRAC = 0.012    # max drift (fraction of frame width) to still be "stopped"
TRACK_TTL_S = 2.0           # forget a track if unseen for this long
EMIT_INTERVAL_S = 0.5       # how often to push live counts to the dashboard
PROCESS_WIDTH = 960         # frames are downscaled to this width for speed
INCIDENT_TYPE = "Stopped Vehicle"

# Simulated GIS position. Waypoints roughly follow NH-44 (Delhi -> Panipat).
# They are approximate and for demonstration only.
HIGHWAY_NAME = "NH-44 (Delhi - Panipat corridor)"
HIGHWAY_PATH = [
    (28.8560, 77.0930),   # near Singhu border
    (28.9931, 77.0151),   # near Sonipat
    (29.1830, 76.9900),   # near Samalkha
    (29.3909, 76.9635),   # near Panipat
]
CAMERA_T = 0.45             # camera position along the path (0 = start, 1 = end)
CAMERA_SPAN = 0.03          # ROI top-to-bottom covers this fraction of the path (~2 km)

# Swap the waypoints above for the Delhi-Mumbai Expressway if you prefer, e.g.:
# HIGHWAY_NAME = "Delhi-Mumbai Expressway (Sohna - Dausa)"
# HIGHWAY_PATH = [(28.2480, 77.0680), (27.9000, 76.7500), (27.3000, 76.4500), (26.9000, 76.3400)]

# ----------------------------------------------------------------------------
# Flask / SocketIO setup
# ----------------------------------------------------------------------------
app = Flask(__name__)
# "threading" mode plays nicely with OpenCV / PyTorch (no eventlet monkey-patching).
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

latest_jpeg = None                       # most recent annotated frame (for /video_feed)
latest_payload = None                    # most recent traffic_update (for new clients)
state_lock = threading.Lock()


def utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# GIS helpers: map a pixel position inside the ROI to simulated lat/lon
# ----------------------------------------------------------------------------
_seg_len = [math.dist(HIGHWAY_PATH[i], HIGHWAY_PATH[i + 1]) for i in range(len(HIGHWAY_PATH) - 1)]
_total_len = sum(_seg_len)


def point_on_path(t):
    """Interpolate a (lat, lon) at fraction t (0-1) along HIGHWAY_PATH."""
    t = min(max(t, 0.0), 1.0)
    target = t * _total_len
    for i, seg in enumerate(_seg_len):
        if target <= seg or i == len(_seg_len) - 1:
            f = target / seg if seg else 0.0
            (la1, lo1), (la2, lo2) = HIGHWAY_PATH[i], HIGHWAY_PATH[i + 1]
            return la1 + (la2 - la1) * f, lo1 + (lo2 - lo1) * f
        target -= seg
    return HIGHWAY_PATH[-1]


def pixel_to_geo(cx, cy, w, h):
    """
    Very simple 'camera calibration': the vertical position in the frame becomes
    distance along the highway (top of frame = further ahead), the horizontal
    position becomes a tiny lateral offset so side-by-side incidents don't overlap.
    """
    nx, ny = cx / w, cy / h
    lat, lon = point_on_path(CAMERA_T + (0.5 - ny) * CAMERA_SPAN)
    return round(lat, 6), round(lon + (nx - 0.5) * 0.0002, 6)


CAMERA_LAT, CAMERA_LON = point_on_path(CAMERA_T)


# ----------------------------------------------------------------------------
# Incident engine: ROI filtering, counting, and stationary-vehicle detection
# ----------------------------------------------------------------------------
class IncidentEngine:
    """
    Consumes per-frame detections and decides which vehicles are 'stopped'.

    A detection is a dict: {"track_id": int, "cls": "car|truck|bus", "box": (x1, y1, x2, y2)}
    A vehicle is stopped when its box centre stays within STOP_RADIUS_FRAC * frame_width
    of an anchor point for >= STOP_SECONDS while inside the ROI.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = {}                                   # track_id -> state dict
        self.incidents = {}                                # track_id -> incident dict
        self.unique = {c: set() for c in CLASS_NAMES}      # cumulative unique IDs in ROI
        self._counter = getattr(self, "_counter", 0)       # incident id counter survives loops

    def update(self, dets, t, w, h, roi_poly):
        """Process one frame. `t` is the video clock in seconds."""
        radius = STOP_RADIUS_FRAC * w
        counts = {c: 0 for c in CLASS_NAMES}
        per_track = {}                                     # for drawing the overlay
        new, cleared = [], []

        for d in dets:
            tid, cls = d["track_id"], d["cls"]
            x1, y1, x2, y2 = d["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            inside = cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) >= 0

            st = self.tracks.setdefault(tid, {"anchor": None, "since": t, "seen": t})
            st["seen"] = t

            if not inside:
                # Left the ROI: not our concern any more; drop any open incident.
                st["anchor"] = None
                if tid in self.incidents:
                    cleared.append(self.incidents.pop(tid))
                per_track[tid] = (False, 0.0, False)
                continue

            counts[cls] += 1
            self.unique[cls].add(tid)

            # Anchor logic: if the centre drifts beyond the radius, the vehicle is
            # moving -> restart the stopwatch (and clear any open incident).
            if st["anchor"] is None or math.dist((cx, cy), st["anchor"]) > radius:
                st["anchor"], st["since"] = (cx, cy), t
                if tid in self.incidents:
                    cleared.append(self.incidents.pop(tid))

            dwell = t - st["since"]
            if dwell >= STOP_SECONDS and tid not in self.incidents:
                self._counter += 1
                lat, lon = pixel_to_geo(cx, cy, w, h)
                inc = {
                    "id": f"INC-{self._counter:04d}",
                    "type": INCIDENT_TYPE,
                    "track_id": tid,
                    "vehicle_class": cls,
                    "lat": lat,
                    "lon": lon,
                    "detected_at": utc_iso(),
                }
                self.incidents[tid] = inc
                new.append(inc)

            per_track[tid] = (True, dwell, tid in self.incidents)

        # Forget tracks that disappeared (and close their incidents).
        for tid in [k for k, s in self.tracks.items() if t - s["seen"] > TRACK_TTL_S]:
            del self.tracks[tid]
            if tid in self.incidents:
                cleared.append(self.incidents.pop(tid))

        return {
            "counts": counts,
            "totals": {c: len(s) for c, s in self.unique.items()},
            "active": list(self.incidents.values()),
            "new": new,
            "cleared": cleared,
            "per_track": per_track,
        }


engine = IncidentEngine()


# ----------------------------------------------------------------------------
# Video sources
# ----------------------------------------------------------------------------
class YoloSource:
    """Reads a file / stream / webcam and yields (frame, detections, video_time)."""

    def __init__(self, source):
        from ultralytics import YOLO                       # imported lazily so --demo needs no YOLO
        self.model = YOLO(MODEL_WEIGHTS)
        self.is_file = os.path.isfile(str(source))
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.paced = self.is_file                          # throttle files to real time

    def frames(self):
        idx, t0 = 0, time.monotonic()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:                           # loop the sample video forever
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.model.predictor = None            # resets ByteTrack state
                    idx = 0
                    continue
                time.sleep(0.5)                            # live stream hiccup: retry
                continue

            if frame.shape[1] > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            res = self.model.track(
                frame, persist=True, conf=CONFIDENCE, classes=list(COCO_VEHICLES),
                tracker="bytetrack.yaml", verbose=False,
            )[0]

            dets = []
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().cpu().tolist()
                clss = res.boxes.cls.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().tolist()
                for tid, c, b in zip(ids, clss, boxes):
                    dets.append({"track_id": tid, "cls": COCO_VEHICLES[c], "box": tuple(b)})

            t = idx / self.fps if self.is_file else time.monotonic() - t0
            idx += 1
            yield frame, dets, t


class DemoSource:
    """
    Synthetic highway scene so the whole stack can be demoed with no video and no
    YOLO. Vehicles drive toward the camera; every ~20 s one 'breaks down' in the
    ROI for ~10 s, which triggers a Stopped Vehicle incident.
    """

    W, H = 960, 540
    fps = 20.0
    paced = True

    def __init__(self):
        self.vehicles, self.next_id = [], 1
        self.rng = random.Random(7)

    def _spawn(self):
        cls = self.rng.choices(CLASS_NAMES, weights=[6, 3, 1])[0]
        self.vehicles.append({
            "id": self.next_id, "cls": cls, "p": 0.0, "lane": self.rng.uniform(-0.8, 0.8),
            "speed": self.rng.uniform(0.10, 0.18), "stop_until": None,
        })
        self.next_id += 1

    def frames(self):
        t, last_spawn, next_breakdown = 0.0, -9.0, 6.0
        while True:
            dt = 1.0 / self.fps
            t += dt
            if t - last_spawn > 0.7:
                self._spawn()
                last_spawn = t

            # Pick a victim that is currently mid-road and not already stopped.
            if t >= next_breakdown:
                candidates = [v for v in self.vehicles if 0.45 < v["p"] < 0.7 and v["stop_until"] is None]
                if candidates:
                    victim = self.rng.choice(candidates)
                    victim["stop_until"] = t + 10.0
                    next_breakdown = t + 20.0

            frame = np.full((self.H, self.W, 3), (60, 62, 58), np.uint8)
            top_y = int(self.H * 0.40)
            road = np.array([[int(.04 * self.W), self.H], [int(.96 * self.W), self.H],
                             [int(.60 * self.W), top_y], [int(.40 * self.W), top_y]], np.int32)
            cv2.fillPoly(frame, [road], (88, 88, 90))
            for lane in (-0.33, 0.33):                     # dashed lane markers
                for k in range(8):
                    p0, p1 = k / 8, (k + 0.5) / 8
                    pts = []
                    for p in (p0, p1):
                        y = int(top_y + (self.H - top_y) * p)
                        x = int(self.W * (0.5 + lane * (0.10 + 0.40 * p) * 1.7))
                        pts.append((x, y))
                    cv2.line(frame, pts[0], pts[1], (220, 220, 220), 2)

            dets, keep = [], []
            for v in self.vehicles:
                broken_down = v["stop_until"] is not None and t < v["stop_until"]
                if not broken_down:                        # broken-down vehicles do not advance
                    v["p"] += v["speed"] * dt
                if v["p"] > 1.05:
                    continue
                keep.append(v)
                p = max(v["p"], 0.0)
                cy = top_y + (self.H - top_y) * p
                cx = self.W * (0.5 + v["lane"] * (0.10 + 0.40 * p) * 0.9)
                bw, bh = (24 + 120 * p) * (1.5 if v["cls"] == "truck" else 1.0), (18 + 70 * p) * (1.4 if v["cls"] != "car" else 1.0)
                box = (cx - bw / 2, cy - bh, cx + bw / 2, cy)
                colour = {"car": (200, 130, 60), "truck": (60, 140, 210), "bus": (70, 180, 90)}[v["cls"]]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, -1)
                dets.append({"track_id": v["id"], "cls": v["cls"], "box": box})
            self.vehicles = keep
            yield frame, dets, t


def make_source(source, demo):
    if demo:
        return DemoSource()
    is_url = str(source).startswith(("rtsp://", "http://", "https://"))
    if not (is_url or str(source).isdigit() or os.path.isfile(str(source))):
        print(f"[!] '{source}' not found - falling back to --demo mode.")
        return DemoSource()
    return YoloSource(source)


# ----------------------------------------------------------------------------
# Drawing + payload helpers
# ----------------------------------------------------------------------------
BOX_COLOURS = {"car": (255, 170, 60), "truck": (60, 160, 255), "bus": (90, 220, 120)}  # BGR


def annotate(frame, dets, per_track, roi_poly):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_poly], (255, 200, 0))
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
    cv2.polylines(frame, [roi_poly], True, (0, 200, 255), 2)

    for d in dets:
        inside, dwell, stopped = per_track.get(d["track_id"], (False, 0.0, False))
        x1, y1, x2, y2 = map(int, d["box"])
        colour = (0, 0, 255) if stopped else BOX_COLOURS[d["cls"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if stopped else 2)
        label = f"{d['cls']} #{d['track_id']}"
        if inside and dwell >= 1.0:
            label += f"  {dwell:.0f}s"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return frame


def build_payload(res):
    """The JSON contract between backend and dashboard."""
    active = res["active"]
    return {
        "timestamp": utc_iso(),
        "highway": HIGHWAY_NAME,
        "camera": {"lat": CAMERA_LAT, "lon": CAMERA_LON},
        "counts": res["counts"],                           # vehicles currently inside the ROI
        "totals": res["totals"],                           # cumulative unique vehicles seen
        "incident": active[0]["type"] if active else None, # active incident type (or null)
        "active_incidents": active,
    }


# ----------------------------------------------------------------------------
# Background worker: capture -> detect -> analyse -> emit
# ----------------------------------------------------------------------------
def pipeline(source_arg, demo):
    global latest_jpeg, latest_payload
    src = make_source(source_arg, demo)
    print(f"[+] Source: {type(src).__name__}  |  YOLO weights: {MODEL_WEIGHTS}")

    roi_poly, roi_poly_size, last_emit, last_t = None, None, 0.0, -1.0
    for frame, dets, t in src.frames():
        started = time.perf_counter()

        h, w = frame.shape[:2]
        if roi_poly_size != (w, h):
            roi_poly = np.array([(int(x * w), int(y * h)) for x, y in ROI_NORM], np.int32)
            roi_poly_size = (w, h)

        if t < last_t:                                     # video looped -> fresh state
            engine.reset()
            socketio.emit("incidents_reset")
        last_t = t

        res = engine.update(dets, t, w, h, roi_poly)
        payload = build_payload(res)

        for inc in res["new"]:                             # push alerts immediately
            socketio.emit("incident_alert", {**payload, "alert": inc})
            print(f"[ALERT] {inc['id']} {inc['vehicle_class']} stopped at {inc['lat']}, {inc['lon']}")
        for inc in res["cleared"]:
            socketio.emit("incident_cleared", {"timestamp": utc_iso(), "incident": inc})

        now = time.monotonic()
        if now - last_emit >= EMIT_INTERVAL_S:             # throttled live counters
            socketio.emit("traffic_update", payload)
            last_emit = now

        ok, buf = cv2.imencode(".jpg", annotate(frame, dets, res["per_track"], roi_poly),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        with state_lock:
            latest_payload = payload
            if ok:
                latest_jpeg = buf.tobytes()

        if src.paced:                                      # keep files at their native speed
            time.sleep(max(0.0, 1.0 / src.fps - (time.perf_counter() - started)))


# ----------------------------------------------------------------------------
# Routes & socket handlers
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the annotated frames (shown in the dashboard sidebar)."""
    def gen():
        while True:
            with state_lock:
                jpg = latest_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("connect")
def on_connect():
    """Give a newly opened dashboard everything it needs to draw the current state."""
    with state_lock:
        payload = latest_payload
    emit("state_sync", {                                   # replies to this client only
        "highway": HIGHWAY_NAME,
        "path": HIGHWAY_PATH,
        "camera": {"lat": CAMERA_LAT, "lon": CAMERA_LON},
        "payload": payload,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highway Incident Detection & GIS Command Center")
    parser.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "sample_highway.mp4"),
                        help="video file, RTSP/HTTP URL, or webcam index (default: sample_highway.mp4)")
    parser.add_argument("--demo", action="store_true", help="use a synthetic scene (no video/YOLO needed)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    socketio.start_background_task(pipeline, args.source, args.demo)
    print(f"[+] Dashboard: http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)