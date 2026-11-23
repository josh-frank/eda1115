#!/usr/bin/env python3
"""stream.py: cli.py, but it /streams instead of printing. Data only; the UI is stream.html.

    python3 stream.py                  # serves on :8000
    curl -N http://raspberrypi.local:8000/stream

GET /stream   Server-Sent Events, one JSON frame per sample:
              {"t": secs since start, "v": volts, "g": uS or null}
              g is null when out of range / no contact.
New clients get the last 60 s replayed first, then live frames.
CORS is open (*) so stream.html can live anywhere.
Read-only: no POST, no state. All interpretation happens in the client.
"""
import argparse
import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from smbus2 import SMBus

ADDR = 0x48                # ADS1115 default (ADDR pin to GND)
REG_CONV, REG_CONFIG = 0x00, 0x01

# AIN0 vs GND, +/-4.096 V, continuous, 128 SPS, comparator off
CONFIG = 0x4000 | 0x0200 | 0x0000 | 0x0080 | 0x0003
LSB = 4.096 / 32768        # volts per count at this gain

R_REF, VCC = 100_000, 3.3

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="0.0.0.0", help="bind address (default all interfaces)")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--hz", type=float, default=10.0, help="samples per second (default 10)")
ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
ap.add_argument("--addr", type=lambda s: int(s, 0), default=ADDR,
                help="ADS1115 address (default 0x48)")
args = ap.parse_args()

HISTORY_S = 60
history = deque(maxlen=int(args.hz * HISTORY_S) + 1)   # encoded SSE frames
clients = set()                                        # one small queue per connection
lock = threading.Lock()


def publish(frame):
    msg = f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode()
    with lock:
        history.append(msg)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:                         # slow client: drop its oldest frame
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass


def reader():
    t0 = time.monotonic()
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
            publish({"t": round(time.monotonic() - t0, 3), "v": round(v, 4),
                     "g": None if g is None else round(g, 4)})
            time.sleep(1 / args.hz)
        except OSError:                                # I2C hiccup: report, retry
            if bus:
                bus.close()
            bus = None
            publish({"t": round(time.monotonic() - t0, 3), "v": None, "g": None})
            time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self.stream()
        else:
            self.send_error(404)

    def stream(self):
        q = queue.Queue(maxsize=200)
        with lock:                                     # snapshot + subscribe atomically
            backfill = list(history)
            clients.add(q)
        try:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for msg in backfill:
                self.wfile.write(msg)
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


threading.Thread(target=reader, daemon=True).start()
srv = ThreadingHTTPServer((args.host, args.port), Handler)
srv.daemon_threads = True
print(f"serving on http://{args.host}:{args.port}  (/stream for SSE)")
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass