#!/usr/bin/env python3
"""elron1115-server.py: cli.py, but it /streams instead of printing. Data + recording; the UI is stream.html.

    python3 stream.py                  # serves on :8000, sessions go to ./sessions
    curl -N http://raspberrypi.local:8000/stream

GET  /stream        Server-Sent Events. Default events are samples:
                        {"t": secs since start, "v": volts, "g": uS or null}
                    (g is null when out of range / no contact). New clients get the
                    last 60 s replayed first, then live frames.
                    A second event type, "rec", reports recording state:
                        {"on": true, "since": t}  or  {"on": false, "saved": "x.srt", "cues": n}
                    It is sent on connect and whenever recording starts or stops.
                    A third, "state", reports the detector's state whenever it changes
                    (and once on connect):  {"state": "IDLE"|"ARM"|"READ", "t": secs}
                    A fourth, "ev", carries each finished detector Event (see
                    elron2.Event.as_dict) when its READ closes. The last 60 s of them are
                    replayed on connect. Events are never written to recordings.
                    The sample frames are unchanged; clients that don't listen for
                    "state" or "ev" never notice them.
POST /record/start  start recording to <outdir>/<timestamp>.srt   (idempotent)
POST /record/stop   stop and write the .srt                       (idempotent)

The .srt matches elron's: one cue per sample, text = uS (4 decimals) or OPEN for
no contact. Samples stream to a .raw.csv first, so a power cut loses nothing: any
leftover .raw.csv is converted to .srt the next time this starts.
"""
import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from smbus2 import SMBus

from elron1115 import Detector

ADDR = 0x48                # ADS1115 default (ADDR pin to GND)
REG_CONV, REG_CONFIG = 0x00, 0x01

# AIN0 vs GND, +/-4.096 V, continuous, 128 SPS, comparator off
CONFIG = 0x4000 | 0x0200 | 0x0000 | 0x0080 | 0x0003
LSB = 4.096 / 32768        # volts per count at this gain

R_REF, VCC = 100_000, 3.3

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="0.0.0.0", help="bind address (default all interfaces)")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--hz", type=float, default=20.0, help="samples per second (default 20)")
ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
ap.add_argument("--addr", type=lambda s: int(s, 0), default=ADDR,
                help="ADS1115 address (default 0x48)")
ap.add_argument("--outdir", default="sessions",
                help="where recorded .srt sessions go (default ./sessions)")
ap.add_argument("--origin", default="*",
                help="Access-Control-Allow-Origin value; set to your page's origin to "
                     "stop other web pages from reading the stream or starting recordings")
args = ap.parse_args()

T0 = time.monotonic()


def now():
    """Stream time: seconds since this process started."""
    return time.monotonic() - T0


# ---- session recording ------------------------------------------------------
OPEN = "OPEN"              # cue text for out of range / no contact


def srt_time(ms):
    h, ms = divmod(int(ms), 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def raw_to_srt(raw_path, end_ms=None):
    """'ms,value' raw log -> .srt. Each cue lasts until the next sample; the last
    lasts until end_ms (or one sample period). -> (srt path or None, cue count)."""
    rows = []
    with open(raw_path) as f:
        for line in f:
            ms, _, val = line.rstrip("\n").partition(",")
            if ms.isdigit():                       # skips a torn last line
                rows.append((int(ms), val or OPEN))
    if not rows:
        return None, 0
    srt_path = raw_path[: -len(".raw.csv")] + ".srt"
    tick = round(1000 / args.hz)
    with open(srt_path, "w", encoding="utf-8") as out:
        for i, (ms, val) in enumerate(rows):
            end = rows[i + 1][0] if i + 1 < len(rows) else (end_ms or ms + tick)
            end = max(end, ms + 1)
            out.write(f"{i + 1}\n{srt_time(ms)} --> {srt_time(end)}\n{val}\n\n")
    return srt_path, len(rows)


class Recorder:
    def __init__(self, outdir):
        self.outdir = outdir
        self.lock = threading.Lock()
        self.active = False
        self.f = None
        self.t0 = 0.0

    def recover(self):
        """Convert orphaned raw logs from a crash; -> list of recovered .srt paths."""
        done = []
        if os.path.isdir(self.outdir):
            for name in sorted(os.listdir(self.outdir)):
                if name.endswith(".raw.csv"):
                    raw = os.path.join(self.outdir, name)
                    path, _ = raw_to_srt(raw)
                    os.remove(raw)
                    if path:
                        done.append(path)
        return done

    def state(self):
        return {"on": True, "since": round(self.t0, 3)} if self.active else {"on": False}

    def start(self):
        """-> True if recording began, False if it already was."""
        with self.lock:
            if self.active:
                return False
            os.makedirs(self.outdir, exist_ok=True)
            base = os.path.join(self.outdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            stem, n = base, 1
            while os.path.exists(stem + ".srt") or os.path.exists(stem + ".raw.csv"):
                n += 1
                stem = f"{base}_{n}"
            self.raw_path = stem + ".raw.csv"
            self.f = open(self.raw_path, "w")
            self.t0 = now()
            self.last_flush = time.monotonic()
            self.active = True
            return True

    def add(self, t, g):
        """Called for every sample frame; a no-op unless recording."""
        if not self.active:
            return
        with self.lock:
            if not self.active:
                return
            val = "" if g is None else f"{g:.4f}"
            self.f.write(f"{round((t - self.t0) * 1000)},{val}\n")
            if time.monotonic() - self.last_flush > 1.0:
                self.f.flush()
                self.last_flush = time.monotonic()

    def stop(self):
        """-> None if not recording, else (srt path or None, cue count)."""
        with self.lock:
            if not self.active:
                return None
            self.active = False
            end_ms = round((now() - self.t0) * 1000)
            self.f.close()
            path, n = raw_to_srt(self.raw_path, end_ms)
            os.remove(self.raw_path)
            return path, n


rec = Recorder(args.outdir)

det = Detector()                                       # fed from publish(), i.e. the reader thread only
cur = {"state": None, "t": 0.0}                        # last state we announced
events = deque(maxlen=50)                              # (t_end, encoded "ev" msg) for late joiners

# ---- fan-out to clients -------------------------------------------------------
HISTORY_S = 60
history = deque(maxlen=int(args.hz * HISTORY_S) + 1)   # encoded SSE frames
clients = set()                                        # one small queue per connection
lock = threading.Lock()


def send_all(msg):
    """Queue msg for every client. Caller holds lock."""
    for q in clients:
        try:
            q.put_nowait(msg)
        except queue.Full:                             # slow client: drop its oldest frame
            try:
                q.get_nowait()
                q.put_nowait(msg)
            except (queue.Empty, queue.Full):
                pass


def named_msg(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def rec_msg(state):
    return named_msg("rec", state)


def state_msg(state, t):
    return named_msg("state", {"state": state, "t": round(t, 3)})


def publish(frame):
    rec.add(frame["t"], frame["g"])
    msg = f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode()
    ev = None
    try:
        ev = det.feed(frame["t"], frame["g"])          # a finished Event, or None
    except Exception as e:                             # a detector bug must never stop the data
        print(f"detector error: {e!r}; resetting it")
        det.reset()
    with lock:
        history.append(msg)
        send_all(msg)
        if det.state != cur["state"]:                  # a transition: tell everyone, once
            cur["state"], cur["t"] = det.state, frame["t"]
            send_all(state_msg(cur["state"], cur["t"]))
        if ev:                                         # a READ just closed
            m = named_msg("ev", ev.as_dict())
            events.append((ev.t_end, m))
            send_all(m)


def announce(state):
    with lock:                                         # state events are not replayed
        send_all(rec_msg(state))


def reader():
    bus = None
    while True:
        try:
            if bus is None:
                bus = SMBus(args.bus)
                bus.write_i2c_block_data(args.addr, REG_CONFIG, [CONFIG >> 8, CONFIG & 0xFF])
                time.sleep(0.02)
            hi, lo = bus.read_i2c_block_data(args.addr, REG_CONV, 2)   # big-endian
            v = int.from_bytes([hi, lo], "big", signed=True) * LSB
            g = (VCC - v) / (v * R_REF) * 1e6 if 0.001 < v < VCC - 0.001 else None
            publish({"t": round(now(), 3), "v": round(v, 4),
                     "g": None if g is None else round(g, 4)})
            time.sleep(1 / args.hz)
        except OSError:                                # I2C hiccup: report, retry
            if bus:
                bus.close()
            bus = None
            publish({"t": round(now(), 3), "v": None, "g": None})
            time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", args.origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):                              # CORS preflight
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/stream":
            self.stream()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/record/start":
            if rec.start():
                announce(rec.state())
            body = rec.state()
        elif self.path == "/record/stop":
            res = rec.stop()
            if res is None:                            # wasn't recording
                body = rec.state()
            else:
                path, n = res
                body = {"on": False, "saved": os.path.basename(path) if path else None, "cues": n}
                announce(body)
        else:
            self.send_error(404)
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream(self):
        q = queue.Queue(maxsize=200)
        with lock:                                     # snapshot + subscribe atomically
            backfill = list(history)
            clients.add(q)
            st = dict(cur)
            recent = list(events)
        try:
            self.send_response(200)
            self.cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for msg in backfill:
                self.wfile.write(msg)
            self.wfile.write(rec_msg(rec.state()))
            if st["state"]:
                self.wfile.write(state_msg(st["state"], st["t"]))
            for t_end, m in recent:                    # recent events, so late joiners see the marks
                if t_end > now() - HISTORY_S:
                    self.wfile.write(m)
            self.wfile.flush()
            while True:
                try:
                    self.wfile.write(q.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")    # keep idle connections alive
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with lock:
                clients.discard(q)


for p in rec.recover():
    print(f"recovered unfinished session -> {p}")
threading.Thread(target=reader, daemon=True).start()
srv = ThreadingHTTPServer((args.host, args.port), Handler)
srv.daemon_threads = True
print(f"serving on http://{args.host}:{args.port}  (/stream for SSE)")
print(f"recordings -> {os.path.abspath(args.outdir)}")
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    res = rec.stop()                                   # don't lose a session on quit
    if res and res[0]:
        print(f"saved {res[0]} ({res[1]} cues)")