# Real-Time Highway Incident Detection & GIS Command Center

A computer-vision prototype that watches highway video, detects **stopped vehicles** in the active lanes, and shows them live on a map-based command dashboard.

YOLOv8 and ByteTrack detect and track vehicles. A small incident engine decides when a vehicle has stopped. Flask-SocketIO streams counts and alerts to a Leaflet.js dashboard, where incidents appear as pulsing red markers.

![Dashboard screenshot](docs/dashboard.png)



## Features

- **Vehicle detection and tracking** for cars, buses and trucks (YOLOv8 + ByteTrack via Ultralytics).
- **Region of Interest (ROI):** only vehicles inside a polygon covering the active lanes are counted and monitored.
- **Stopped Vehicle incidents:** a vehicle whose box center stays within a small distance for more than 5 seconds triggers an incident. The incident clears automatically when the vehicle moves, leaves the ROI, or disappears.
- **Live traffic counters:** current vehicles per class in the ROI, plus cumulative unique vehicles since start.
- **GIS dashboard:** Leaflet.js + OpenStreetMap map of an NH-44 segment, pulsing incident markers with popups, and a live event log (click an alert to zoom to it).
- **Annotated camera feed** (MJPEG) in the sidebar, which makes it easy to tune the ROI.
- **State sync:** a dashboard opened mid-incident immediately shows the current counts and active incidents.
- **Demo mode:** a synthetic highway scene so the whole system runs without a video file or YOLO.

## Architecture

```mermaid
flowchart LR
    A[Video file / RTSP / webcam] --> B[YOLOv8 + ByteTrack]
    B --> C[ROI filter + stationary-vehicle engine]
    C --> D[Flask-SocketIO]
    C --> E[Annotated MJPEG feed]
    D --> F[Leaflet.js dashboard]
    E --> F
```

## Tech stack

Python, YOLOv8 (Ultralytics), ByteTrack, OpenCV, NumPy, Flask, Flask-SocketIO, Leaflet.js, OpenStreetMap, HTML/CSS/JS.

## Project structure

```
highway-incident-command-center/
├── app.py              # Backend: detection, incident engine, SocketIO events
├── requirements.txt
├── templates/
│   └── index.html      # Leaflet + Socket.IO dashboard
└── README.md
```

## Setup

Requires **Python 3.10 to 3.12**.

```bash
git clone https://github.com/tanu-dev0/highway-incident-command-center.git
cd highway-incident-command-center

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

The first run with a video downloads the `yolov8n.pt` weights automatically (internet required). The dashboard also loads Leaflet, map tiles and fonts from CDNs, so it needs an internet connection.

## Usage

```bash
python app.py --demo                       # synthetic scene, no video needed
python app.py --source sample_highway.mp4  # video file
python app.py --source rtsp://user:pass@host/stream
python app.py --source 0                   # webcam
```

Then open **http://127.0.0.1:5000**.

| Option | Description |
|---|---|
| `--source` | Video file, RTSP/HTTP URL or webcam index (default `sample_highway.mp4`) |
| `--demo` | Use the synthetic scene |
| `--host`, `--port` | Server address (default `127.0.0.1:5000`) |

If the source file does not exist, the app falls back to demo mode. Use footage from a **fixed camera** (for example an overpass), because camera movement breaks stationary-vehicle detection.

## Configuration

All settings are constants at the top of `app.py`.

| Setting | Default | Meaning |
|---|---|---|
| `ROI_NORM` | trapezoid | Active-lane polygon in normalized (0 to 1) coordinates. Tune this per camera. |
| `STOP_SECONDS` | `5.0` | Time a vehicle must stay put to raise an incident |
| `STOP_RADIUS_FRAC` | `0.012` | Max drift (fraction of frame width) still counted as stationary |
| `CONFIDENCE` | `0.35` | YOLO detection confidence threshold |
| `MODEL_WEIGHTS` | `yolov8n.pt` | Any Ultralytics model; larger models are more accurate but slower |
| `HIGHWAY_PATH`, `CAMERA_T` | NH-44 waypoints | Simulated geography used for incident coordinates |

## How stopped-vehicle detection works

1. YOLOv8 detects cars, buses and trucks, and ByteTrack assigns each a persistent track ID.
2. A vehicle counts only if its box center is inside the ROI polygon.
3. Each tracked vehicle gets an **anchor point**. If its center moves farther than the radius from the anchor, the anchor resets and the timer restarts.
4. When the time since the anchor was set reaches `STOP_SECONDS`, a **Stopped Vehicle** incident is raised.
5. The pixel position is mapped to a simulated latitude/longitude along the highway, and an alert is pushed to the dashboard.

## Socket events

| Event | When | Contents |
|---|---|---|
| `state_sync` | On connect | Highway path, camera position, latest state |
| `traffic_update` | About twice per second | Counts, totals, active incident type, timestamp, active incidents |
| `incident_alert` | A new incident is raised | The update payload plus the `alert` (id, type, vehicle class, lat/lon) |
| `incident_cleared` | An incident ends | The cleared incident |
| `incidents_reset` | The sample video loops | Tells the dashboard to clear markers |

Example `traffic_update`:

```json
{
  "timestamp": "2026-10-15T07:13:46+00:00",
  "highway": "NH-44 (Delhi - Panipat corridor)",
  "camera": { "lat": 29.0861, "lon": 77.0028 },
  "counts": { "car": 8, "bus": 2, "truck": 0 },
  "totals": { "car": 25, "bus": 9, "truck": 0 },
  "incident": null,
  "active_incidents": []
}
```

## Limitations

- **Coordinates are simulated.** Pixel position is mapped to a point along approximate NH-44 waypoints. It is not a real camera calibration or geo-referencing.
- If heavy occlusion makes the tracker assign a new ID to a stopped vehicle, its timer restarts.
- Detection quality depends on camera angle, lighting and the YOLO model size.
- Incidents are held in memory only, so they are lost when the server restarts.

## Future work

- Persist incidents (SQLite or PostgreSQL) for history and analytics.
- Real camera calibration (homography) for accurate coordinates.
- Additional incident types such as wrong-way driving, sudden congestion and pedestrians on the carriageway.
- Multiple cameras on one map.

## Author

**Tanu Sharma**: [GitHub](https://github.com/tanu-dev0)