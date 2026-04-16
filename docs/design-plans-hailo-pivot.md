# Design Plans for TeknoFest Fighter UAV System

## The Critical Pivot: Hailo-8L, Not NVIDIA

Gemini is right. My previous response assumed NVIDIA GPU + TensorRT. Your actual stack is **RPi 5 + Hailo-8L**. This changes the inference pipeline fundamentally:

| Aspect | Old (TensorRT) | New (Hailo-8L) |
|--------|----------------|-----------------|
| Model format | `.engine` (TensorRT) | `.hef` (Hailo Executable Format) |
| Compiler | `trtexec` | Hailo Dataflow Compiler (DFC) |
| Runtime API | TensorRT C++ API | HailoRT C++/Python API |
| Pre/post-processing | GPU (CUDA) | **CPU (RPi5 ARM cores)** |
| Quantization | FP16 | **INT8 only** (Hailo does not do FP16) |
| Throughput | ~100+ FPS typical | ~30-60 FPS for YOLOv8s-equivalent |

**INT8 quantization** means you need a calibration dataset during compilation. This is an extra step but Hailo's INT8 is well-optimized — 13 TOPS at INT8 is competitive.

---

## Design A: "Python-First" (Fastest to Build)

**Philosophy:** Everything in Python. Maximize development speed. Use Hailo's Python bindings everywhere.

```
┌─────────────────────── AIRCRAFT (RPi 5) ───────────────────────┐
│                                                                  │
│  libcamera (Python picamera2)                                    │
│       │                                                          │
│       ▼                                                          │
│  [Frame Queue] ──► HailoRT Python ──► YOLOv11s (.hef)          │
│       │                    │                                     │
│       │                    ▼                                     │
│       │            Post-process (NumPy)                          │
│       │                    │                                     │
│       │                    ├──► ByteTrack (Python)               │
│       │                    │         │                            │
│       │                    │         ▼                            │
│       │                    │    Lock State Machine                │
│       │                    │         │                            │
│       │                    ▼         ▼                            │
│       │              QR Detector   MAVLink (pymavlink)           │
│       │              (OpenCV)      → Pixhawk (UART)             │
│       │                                                          │
│       ▼                                                          │
│  Video Recorder (OpenCV VideoWriter, H264)                      │
│                                                                  │
│  MAVLink Telemetry ◄──► Pixhawk Cube Orange                    │
│       │                                                          │
│       ▼ (RFD868x)                                               │
└───────┼──────────────────────────────────────────────────────────┘
        │
   Radio Link
        │
┌───────┼──────────────── GROUND (Laptop) ────────────────────────┐
│       ▼                                                          │
│  GCS Application (Python)                                        │
│  ┌─────────────────────────────────────────┐                    │
│  │  FastAPI backend                         │                    │
│  │  ├── MAVLink listener (pymavlink)       │                    │
│  │  ├── Competition server API client      │                    │
│  │  │   (requests/httpx)                   │                    │
│  │  ├── Lock/Kamikaze packet forwarder     │                    │
│  │  └── WebSocket → frontend               │                    │
│  ├─────────────────────────────────────────┤                    │
│  │  React/Leaflet frontend (browser)       │                    │
│  │  ├── Map (boundaries, rivals, ADS)      │                    │
│  │  ├── Telemetry dashboard                │                    │
│  │  ├── Lock status + timer                │                    │
│  │  └── Server time display                │                    │
│  └─────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Fastest development time — picamera2, pymavlink, OpenCV, FastAPI all have excellent Python APIs
- Hailo provides official `hailo-apps-infra` Python examples with YOLO pipelines already working
- Easy to debug and iterate
- Single language across entire stack

**Cons:**
- Python GIL limits true parallelism — pre/post-processing on ARM cores may bottleneck at ~15-20 FPS
- Higher latency per frame vs C++
- `picamera2` + HailoRT Python may have overhead that eats into your frame budget

**Risk:** If you need >20 FPS, Python may not deliver on RPi5. But 15 FPS is the spec minimum for video, and lock detection doesn't need blazing speed — it needs reliability over 4 seconds.

---

## Design B: "C++ Core, Python GCS" (Best Performance)

**Philosophy:** C++ for everything latency-critical on the aircraft. Python for the GCS where latency doesn't matter.

```
┌─────────────────────── AIRCRAFT (RPi 5) ───────────────────────┐
│                                                                  │
│  C++ Pipeline (single process, multi-threaded)                  │
│                                                                  │
│  Thread 1: Camera Grabber                                        │
│    libcamera C++ API → ring buffer                              │
│                                                                  │
│  Thread 2: Inference                                             │
│    dequeue frame → preprocess (OpenCV C++) →                    │
│    HailoRT C++ async infer → postprocess (NMS, decode boxes)   │
│                                                                  │
│  Thread 3: Tracker + Logic                                       │
│    ByteTrack C++ → Lock State Machine →                         │
│    lock/kamikaze event queue                                    │
│                                                                  │
│  Thread 4: QR Decoder                                            │
│    OpenCV QRCodeDetector (activated in kamikaze mode only)      │
│                                                                  │
│  Thread 5: MAVLink Comms                                         │
│    MAVSDK C++ → Pixhawk UART                                    │
│    Sends: guidance vectors, mode changes                        │
│    Receives: GPS, altitude, speed, battery, attitude            │
│                                                                  │
│  Thread 6: Video Recorder                                        │
│    OpenCV VideoWriter → H264/MP4 with timestamp overlay         │
│                                                                  │
│  Shared state: lock-free queues between threads                 │
│                                                                  │
│  Telemetry out → RFD868x → GCS                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────── GROUND (Laptop) ────────────────────────┐
│                                                                  │
│  GCS (Python — same as Design A)                                │
│  FastAPI + Leaflet/React                                        │
│  pymavlink ← RFD868x                                           │
│  Competition server API (requests/httpx)                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Maximum FPS on RPi5 — C++ pre/post-processing is 3-5x faster than NumPy on ARM
- True multi-threading (no GIL)
- HailoRT C++ API gives you async inference with zero-copy buffer passing
- libcamera C++ avoids picamera2 overhead
- GCS stays in Python where dev speed matters and latency doesn't

**Cons:**
- Significantly more development time for the aircraft pipeline
- Harder to debug on-device
- MAVSDK C++ is less documented than pymavlink
- Cross-compiling for RPi5 aarch64 adds build complexity

**Risk:** Development time. If your team is stronger in Python, the C++ pipeline could take 2-3x longer to build and debug.

---

## Design C: "Hailo TAPPAS Pipeline + Python Logic" (Hybrid, Recommended)

**Philosophy:** Use Hailo's GStreamer-based TAPPAS framework for the video pipeline (camera → inference → display), write application logic in Python on top.

```
┌─────────────────────── AIRCRAFT (RPi 5) ───────────────────────┐
│                                                                  │
│  GStreamer Pipeline (managed by TAPPAS / hailo-apps)            │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ libcamerasrc → videoconvert → hailonet (.hef) →     │       │
│  │ hailofilter (postprocess) → appsink                  │       │
│  └──────────────────────┬──────────────────────────────┘       │
│                          │                                       │
│                    Python callback                               │
│                          │                                       │
│                          ▼                                       │
│              ┌──── App Logic (Python) ────┐                     │
│              │                             │                     │
│              │  ByteTrack tracker          │                     │
│              │  Lock State Machine         │                     │
│              │  QR decoder (OpenCV)        │                     │
│              │  MAVLink (pymavlink)        │                     │
│              │  Video recorder             │                     │
│              │  Telemetry packager         │                     │
│              └─────────────┬───────────────┘                    │
│                             │                                    │
│                     Pixhawk (UART)                               │
│                     RFD868x (UART)                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────── GROUND (Laptop) ────────────────────────┐
│  Same Python GCS as other designs                                │
└──────────────────────────────────────────────────────────────────┘
```

**How TAPPAS works:** Hailo provides GStreamer elements (`hailonet`, `hailofilter`, `hailooverlay`) that handle the entire camera→inference→postprocess chain in C/C++ under the hood, but expose results to your Python code via GStreamer appsink callbacks. You get C++-speed inference with Python-level development ease.

**Pros:**
- Hailo's own recommended approach — best supported, most examples available
- Camera capture + inference + postprocess all run in optimized C/C++ GStreamer elements
- Your Python code only handles application logic (tracking, lock FSM, comms) — the light stuff
- Hailo's `rpicam-apps` fork has working RPi Camera Module 3 + Hailo-8L examples out of the box
- Avoids writing any C++ yourself while getting near-C++ inference performance

**Cons:**
- GStreamer pipeline debugging is its own skill — pipeline stalls can be opaque
- Less control over frame timing than a custom C++ pipeline
- TAPPAS is Hailo-specific — knowledge doesn't transfer to other platforms

**Risk:** GStreamer learning curve. But Hailo's examples for RPi5 are solid starting points.

---

## GCS Design Options

For the GCS specifically, two approaches:

### GCS Option 1: Python Web App (FastAPI + Leaflet)
```
Browser ←WebSocket→ FastAPI ←pymavlink→ RFD868x
                       ↕
              Competition Server (HTTP)
```
- Map: Leaflet.js with OpenStreetMap tiles (or offline tiles)
- UI: HTML/CSS/JS in browser, served by FastAPI
- Comms: WebSocket for real-time updates to map
- **Best if:** you want full control, clean separation, easy to demo on any screen

### GCS Option 2: PyQt Desktop App
```
PyQt Window (map widget + telemetry panels)
    ├── pymavlink thread
    ├── server API thread
    └── map (QWebEngineView + Leaflet, or matplotlib)
```
- Single executable, no browser needed
- **Best if:** you want a self-contained app that looks "professional" for technical control

### GCS Option 3: QGroundControl + Custom Plugin
- Fork QGC, add custom widget for competition server API
- **Best if:** you want autopilot configuration UI for free
- **Worst if:** you don't want to learn QGC's C++/Qt codebase

**My recommendation:** GCS Option 1 (FastAPI + Leaflet). Fastest to build, easiest to debug, judges see a clean map in a browser, and your team can split frontend/backend work.

---

## My Recommendation: Design C + GCS Option 1

| Component | Choice | Why |
|-----------|--------|-----|
| Aircraft inference | TAPPAS/GStreamer + HailoRT | Hailo's own optimized path, proven on RPi5 |
| Aircraft logic | Python (tracker, FSM, comms) | Fast development, adequate performance for logic |
| Model export | PyTorch → ONNX → HEF (Hailo DFC) | Only path available for Hailo-8L |
| Quantization | INT8 with calibration set | Only option on Hailo; use ~200 representative images |
| GCS | FastAPI + Leaflet.js | Fastest to build, clean for demo |
| MAVLink | pymavlink (both aircraft + GCS) | Mature, well-documented, ArduPilot native |

### Execution Order (corrected for Hailo):

1. **GCS first** — pass technical control or you don't fly
2. **Dataset fix** — combine splits, 80/10/10, investigate orphans
3. **Train YOLOv11s+P2** — on your desktop GPU, Ultralytics
4. **Export to HEF** — Hailo DFC on x86 machine, INT8 calibration
5. **Aircraft pipeline** — TAPPAS GStreamer + Python logic on RPi5
6. **Integration** — vision → MAVLink → Pixhawk → GCS → server
7. **Flight test**
