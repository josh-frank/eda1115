#!/usr/bin/env python3
"""
elron.py: electrodermography live read-out.  Three states, no interpretation.

    IDLE   no contact, or the signal is still settling. No events are emitted.
    ARM    watching. Any change bigger and faster than the thresholds opens a READ.
    READ   one monotone excursion is open. It closes when the signal gives back
           half of it, stops making progress, or times out. Each READ becomes
           one logged event.

Elron only measures: direction (+ conductance up / - conductance down),
amplitude in %, time to peak, and peak speed in %/s. What an event MEANS is up to
the human. (A needle "fall" is conductance DOWN, i.e. resistance UP.)

Everything works in log-conductance, so thresholds are percent changes and the
same settings serve a dry day and a sweaty one.

An excursion ends at its extreme, and the next one can start right there. So a
held squeeze is two events (rise, then a fast opposite event that starts at the
peak). Events that start at the previous event's peak, in the opposite
direction, are tagged `rebound` (structure only; hide them with `h`).

Sources:   live ADS1115 (smbus2)  |  --sim  |  --replay FILE.srt
Replay:    headless by default: runs the detector over a recording and prints
           an event table, scored against any "[...] READ" label cues in the file.
           Add --play [SPEED] to watch it in the curses UI instead.
Tuning:    --set start_pct=8 --set min_rate_pct_s=2   (see Params)
Self-test: --selftest   (synthetic read + squeeze, no hardware)

Keys:  q quit   r record .srt   m mark a read (type a comment, Enter)
       s set tone arm here   f follow slow baseline   k/l (or down/up) needle range
       h show/hide rebound events
"""
import argparse
import bisect
import curses
import locale
import math
import os
import queue
import random
import re
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime

IDLE, ARM, READ = "IDLE", "ARM", "READ"
GLYPH = {-1: "▼", +1: "▲"}          # conductance down / up; the words are up to you


# --- tunables ----------------------------------------------------------------
@dataclass
class Params:
    smooth_tau_s: float = 0.15      # tiny input filter
    settle_s: float = 3.0           # signal must be flat this long before arming
    flat_pct: float = 2.0           # "flat" = within this many % over settle_s
    start_pct: float = 6.0          # drawdown/drawup that opens a READ
    min_amp_pct: float = 8.0        # smaller events are discarded
    min_rate_pct_s: float = 1.0     # must be moving at least this fast (drift is ~0.25)
    onset_window_s: float = 12.0    # how far back to look for where a change began
    reversal_frac: float = 0.5      # READ closes when this share of it is given back
    stall_s: float = 3.0            # a READ closes when it makes no real progress for this long
    read_timeout_s: float = 30.0    # a READ can't stay open longer than this
    slow_tau_s: float = 60.0        # slow baseline, logged/displayed as context only
    rate_win_s: float = 0.5         # window for the peak-speed measurement
    gap_s: float = 1.0              # a hole in the data longer than this = lost contact
    rebound_gap_s: float = 1.0      # "starts at the previous peak" tolerance


HIST_S = 60.0


# --- the detector (pure logic: no curses, no I/O) -----------------------------
@dataclass
class Event:
    direction: int
    t_start: float
    t_peak: float
    t_end: float
    pre_g: float
    peak_g: float
    amp_pct: float
    rate_pct_s: float
    rebound: bool = False

    @property
    def duration(self):
        return self.t_peak - self.t_start

    def cue(self):
        return (f"EVENT {self.amp_pct:+.1f}% {self.duration:.1f}s "
                f"{self.rate_pct_s:.1f}%/s" + (" rebound" if self.rebound else ""))


def _x_at(pts, tq):
    """Smoothed log-conductance at (or just before) time tq."""
    i = bisect.bisect_right(pts, (tq, float("inf"))) - 1
    return pts[max(i, 0)][1]


class Detector:
    def __init__(self, params=None):
        self.p = params or Params()
        self.slow = None
        self.aborted = 0            # READs thrown away because contact was lost
        self._to_idle()

    def _to_idle(self):
        if getattr(self, "ev", None) is not None:
            self.aborted += 1
        self.state = IDLE
        self.hist = deque()
        self.xs = None
        self.last_t = None
        self.ev = None
        self.floor_t = None
        self.prev = None            # (direction, t_peak) of the last closed event

    # -- public ---------------------------------------------------------------
    def feed(self, t, g):
        """One sample (seconds, µS). g=None means no contact. -> finished Event or None."""
        p = self.p
        if g is None or g <= 0:
            self._to_idle()
            return None
        if self.last_t is not None and t - self.last_t > p.gap_s:
            self._to_idle()
        dt = 0.0 if self.last_t is None else max(t - self.last_t, 1e-6)

        x = math.log(g)
        if self.xs is None:
            self.xs = x
            if self.slow is None:
                self.slow = x
        else:
            self.xs += (x - self.xs) * (1 - math.exp(-dt / p.smooth_tau_s))
            self.slow += (self.xs - self.slow) * (1 - math.exp(-dt / p.slow_tau_s))
        self.last_t = t
        self.hist.append((t, self.xs))
        while self.hist and t - self.hist[0][0] > HIST_S:
            self.hist.popleft()

        if self.state == IDLE:
            if self._flat(t):
                self.state = ARM
                self.floor_t = t - self.p.settle_s      # never backdate into the settling period
            return None

        pts = list(self.hist)
        if self.state == ARM:
            self._maybe_open(t, pts)
            return None
        return self._update_read(t, pts)

    # -- internals ------------------------------------------------------------
    def _flat(self, t):
        p = self.p
        if t - self.hist[0][0] < p.settle_s:
            return False
        xs = [x for tt, x in self.hist if tt >= t - p.settle_s]
        return max(xs) - min(xs) <= math.log1p(p.flat_pct / 100)

    def _maybe_open(self, t, pts):
        p = self.p
        lo_t = max(t - p.onset_window_s, self.floor_t)
        win = [a for a in pts if a[0] >= lo_t]
        if len(win) < 2:
            return False
        hi = max(reversed(win), key=lambda a: a[1])     # latest max / min
        lo = min(reversed(win), key=lambda a: a[1])
        x = self.xs
        start = math.log1p(p.start_pct / 100)
        speed = (x - _x_at(pts, t - 1.0)) / 1.0
        floor_rate = math.log1p(p.min_rate_pct_s / 100)
        cands = []
        if x - hi[1] <= -start and speed <= -floor_rate:
            cands.append((-1, hi, hi[1] - x))
        if x - lo[1] >= start and speed >= floor_rate:
            cands.append((+1, lo, x - lo[1]))
        if not cands:
            return False
        d, (ot, ox), _ = max(cands, key=lambda c: c[2])

        since = [a for a in pts if a[0] >= ot]
        pk_t, pk_x = (min if d < 0 else max)(reversed(since), key=lambda a: a[1])
        maxr = 0.0
        for tt, xx in since:
            maxr = max(maxr, d * (xx - _x_at(pts, tt - p.rate_win_s)) / p.rate_win_s)
        rebound = bool(self.prev and d == -self.prev[0]
                       and abs(ot - self.prev[1]) <= p.rebound_gap_s)
        self.ev = dict(d=d, t0=ot, pre=ox, pk_t=pk_t, pk_x=pk_x, maxr=maxr, rebound=rebound)
        self.state = READ
        return True

    def _update_read(self, t, pts):
        p, e = self.p, self.ev
        d = e["d"]
        if d * (self.xs - e["pk_x"]) > 0:
            e["pk_x"], e["pk_t"] = self.xs, t
        r = d * (self.xs - _x_at(pts, t - p.rate_win_s)) / p.rate_win_s
        e["maxr"] = max(e["maxr"], r)
        excursion = d * (e["pk_x"] - e["pre"])
        reversal = -d * (self.xs - e["pk_x"])
        if reversal >= p.reversal_frac * excursion:
            return self._close(t, pts, reversed_=True)
        if t - e["t0"] >= p.stall_s:
            floor_rate = math.log1p(p.min_rate_pct_s / 100)
            progress = d * (self.xs - _x_at(pts, t - p.stall_s))
            if progress < 0.5 * floor_rate * p.stall_s:
                return self._close(t, pts, reversed_=False)
        if t - e["t0"] >= p.read_timeout_s:
            return self._close(t, pts, reversed_=False)
        return None

    def flush(self):
        """End of data: close any open READ. -> Event or None."""
        if self.state != READ or self.last_t is None:
            return None
        return self._close(self.last_t, list(self.hist), reversed_=False)

    def _close(self, t, pts, reversed_):
        p, e = self.p, self.ev
        d = e["d"]
        pre_g, peak_g = math.exp(e["pre"]), math.exp(e["pk_x"])
        t10 = e["t0"]                                   # where it first leaves 10% of the way
        for tt, xx in pts:
            if tt >= e["t0"] and d * (xx - e["pre"]) >= 0.1 * d * (e["pk_x"] - e["pre"]):
                t10 = tt
                break
        ev = Event(direction=d, t_start=min(t10, e["pk_t"]), t_peak=e["pk_t"], t_end=t,
                   pre_g=pre_g, peak_g=peak_g,
                   amp_pct=(peak_g / pre_g - 1) * 100,
                   rate_pct_s=abs(math.exp(d * e["maxr"]) - 1) * 100,
                   rebound=e["rebound"])
        self.ev = None
        self.state = ARM
        if reversed_:
            self.floor_t, self.prev = e["pk_t"], (d, e["pk_t"])
            self._maybe_open(t, pts)        # the opposite excursion may start right here
        else:
            self.floor_t, self.prev = t, None
        return ev if abs(ev.amp_pct) >= p.min_amp_pct else None


# --- synthetic signal (live --sim and --selftest) -------------------------------
def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _read_shape(s, depth, tf, tr):
    if s < 0:
        return 0.0
    return -depth * _ease(s / tf) if s < tf else -depth * math.exp(-(s - tf) / tr)


def _squeeze_shape(s, amp, hold):
    if s < 0:
        return 0.0
    if s < 0.4:
        return amp * _ease(s / 0.4)
    top = amp * (1 - 0.03 * min(s - 0.4, hold))
    if s < 0.4 + hold:
        return top
    s2 = s - 0.4 - hold
    if s2 < 0.3:
        return top + (-0.06 - top) * _ease(s2 / 0.3)       # cliff, then undershoot
    return -0.06 * math.exp(-(s2 - 0.3) / 3.0)


def sim_trace(t, events, base=6.0):
    """events: [(kind, t0, a, b, c)] -> conductance in µS at time t."""
    ln = 0.03 * math.sin(t / 40.0)
    for kind, t0, a, b, c in events:
        if kind == "read":
            ln += _read_shape(t - t0, a, b, c)
        else:
            ln += _squeeze_shape(t - t0, a, b)
    return base * math.exp(ln + random.gauss(0, 0.0015))


def random_event(t):
    if random.random() < 0.75:
        return ("read", t, random.uniform(0.12, 0.35), random.uniform(2, 5), random.uniform(4, 10))
    return ("squeeze", t, random.uniform(0.3, 0.5), random.uniform(1, 3), 0)


# --- sources (each pushes (t_seconds, g_uS | None) into a queue) -----------------
ADDR = 0x48
REG_CONV, REG_CONFIG = 0x00, 0x01
PGA = {6.144: 0x0000, 4.096: 0x0200, 2.048: 0x0400, 1.024: 0x0600, 0.512: 0x0800, 0.256: 0x0A00}


def run_i2c(q, stop, bus_no, addr, fsr, r_ref, vcc):
    from smbus2 import SMBus
    cfg = 0x4000 | PGA[fsr] | 0x0000 | 0x0080 | 0x0003     # AIN0/GND, continuous, 128 SPS
    lsb = fsr / 32768
    with SMBus(bus_no) as bus:
        bus.write_i2c_block_data(addr, REG_CONFIG, [cfg >> 8, cfg & 0xFF])
        time.sleep(0.02)
        while not stop.is_set():
            hi, lo = bus.read_i2c_block_data(addr, REG_CONV, 2)
            v = int.from_bytes([hi, lo], "big", signed=True) * lsb
            t = time.monotonic()
            if 0.001 < v < min(vcc - 0.001, fsr * 0.98):
                q.put((t, (vcc - v) / (v * r_ref) * 1e6))
            else:
                q.put((t, None))                         # rail / full-scale: no usable reading
            time.sleep(0.02)


def run_sim(q, stop):
    t0, events, nxt = time.monotonic(), [], 6.0
    while not stop.is_set():
        now = time.monotonic()
        t = now - t0
        if t >= nxt:
            events.append(random_event(t))
            events = events[-6:]
            nxt = t + random.uniform(10, 30)
        q.put((now, sim_trace(t, events)))
        time.sleep(0.02)


def run_play(q, stop, samples, speed):
    """Replay a recording into the UI at `speed`x. Timestamps stay in file time."""
    prev = None
    for t, g in samples:
        if stop.is_set():
            return
        if prev is not None:
            time.sleep(max(t - prev, 0) / speed)
        prev = t
        q.put((t, g))


def reader_thread(q, stop, fn, *args):
    try:
        fn(q, stop, *args)
    except Exception as e:                      # surface errors in the UI
        q.put(f"{type(e).__name__}: {e}")


# --- .srt reading (replay) -------------------------------------------------------
def _srt_secs(h, m, s, ms):
    return ((int(h) * 60 + int(m)) * 60 + int(s)) + int(ms) / 1000


def load_srt(path, flip=False):
    """-> (samples [(t, g|None)], labels [(start, end, text)]).
    Numeric cues are samples, 'OPEN' is no contact, other text cues are labels."""
    samples, labels = [], []
    with open(path, encoding="utf-8") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            continue
        a = m.groups()
        start, end = _srt_secs(*a[:4]), _srt_secs(*a[4:])
        text = " ".join(lines[2:]).strip()
        try:
            samples.append((start, float(text)))
        except ValueError:
            if text == "OPEN":
                samples.append((start, None))
            else:
                labels.append((start, end, text))
    samples.sort(key=lambda s: s[0])
    if flip:
        lg = [math.log(g) for _, g in samples if g]
        mid = statistics.median(lg)
        samples = [(t, math.exp(2 * mid - math.log(g)) if g else None) for t, g in samples]
    return samples, labels


# --- headless replay + scoring ------------------------------------------------------
def run_events(samples, params):
    det, out = Detector(params), []
    first_arm = None
    for t, g in samples:
        ev = det.feed(t, g)
        if first_arm is None and det.state != IDLE:
            first_arm = t
        if ev:
            out.append(ev)
    last = det.flush()
    if last:
        out.append(last)
    return out, det, first_arm


def print_table(events, labels, show_rebounds=True):
    reads = [(a, b, tx) for a, b, tx in labels if tx.rstrip().upper().endswith("READ")]
    tol = 3.0

    def hit(ev):
        return [i for i, (a, b, _) in enumerate(reads)
                if ev.t_start <= b + tol and ev.t_peak >= a - tol]

    print(f"{'#':>3} {'start':>8} {'dir':>3} {'amp%':>8} {'dur s':>7} {'peak %/s':>9}  note")
    for n, ev in enumerate(events, 1):
        if ev.rebound and not show_rebounds:
            continue
        tags = []
        if ev.rebound:
            tags.append("rebound")
        h = hit(ev)
        if h:
            tags.append("in label " + ",".join(str(i + 1) for i in h))
        print(f"{n:>3} {ev.t_start:8.1f} {GLYPH[ev.direction]:>3} {ev.amp_pct:+8.1f} "
              f"{ev.duration:7.1f} {ev.rate_pct_s:9.1f}  {' '.join(tags)}")
    return reads, hit


def replay_report(path, params, flip, show_rebounds=True):
    samples, labels = load_srt(path, flip)
    if not samples:
        print("no numeric cues found")
        return
    events, det, first_arm = run_events(samples, params)
    span = samples[-1][0] - samples[0][0]
    print(f"{path}: {len(samples)} samples, {span:.0f} s, first ARM at "
          f"{(first_arm or 0):.1f} s, {len(events)} events, {det.aborted} aborted")
    reads, hit = print_table(events, labels, show_rebounds)
    if not reads:
        return
    print("\nlabeled reads (largest event overlapping each window):")
    hits = 0
    for i, (a, b, tx) in enumerate(reads):
        mine = [e for e in events if i in hit(e)]
        hits += bool(mine)
        best = max(mine, key=lambda e: abs(e.amp_pct), default=None)
        what = (f"{GLYPH[best.direction]} {best.amp_pct:+.1f}% in {best.duration:.1f}s"
                f"   ({len(mine)} event{'s' if len(mine) != 1 else ''} in window)"
                if best else "MISSED")
        print(f"  {i + 1}. {a:6.1f}-{b:6.1f} s  {what}")
    last = max(b for _, b, _ in reads) + 30
    quiet = [e for e in events if e.t_start > last]
    outside = [e for e in events if not hit(e)]
    print(f"\n{hits}/{len(reads)} labeled reads have an event in their window; {len(outside)} events "
          f"outside any label window; {len(quiet)} events after {last:.0f} s (quiet stretch)")


# --- self-test ----------------------------------------------------------------------
def selftest(params):
    random.seed(1)
    sched = [("read", 40, 0.25, 3.0, 6.0), ("squeeze", 90, 0.4, 2.0, 0)]
    samples = [(i / 20, sim_trace(i / 20, sched)) for i in range(140 * 20)]
    events, det, _ = run_events(samples, params)
    print_table(events, [])
    fall = [e for e in events if e.direction < 0 and 38 <= e.t_start <= 46]
    rise = [e for e in events if e.direction > 0 and 88 <= e.t_start <= 92]
    edge = [e for e in events if e.direction < 0 and 90 <= e.t_start <= 96]
    assert fall and abs(fall[0].amp_pct) > 15 and fall[0].rate_pct_s < 20, "read not found"
    assert rise and rise[0].rate_pct_s > 50, "squeeze rise not found / not fast"
    assert edge and edge[0].rate_pct_s > 50, "squeeze release not found / not fast"
    print("\nselftest ok: slow fall found; squeeze = fast rise + fast fall")


# --- session recording (.srt) ---------------------------------------------------------
def srt_time(ms):
    h, ms = divmod(int(ms), 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def raw_to_srt(raw_path, end_ms=None):
    """'S ms value' and 'T ms end text' lines -> .srt. Samples last until the next
    sample; text cues (events, marks) keep their own span. -> (path, cue count)."""
    nums, texts = [], []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            try:
                if p[0] == "S" and len(p) == 3:
                    nums.append((int(p[1]), p[2]))
                elif p[0] == "T" and len(p) == 4:
                    texts.append((int(p[1]), int(p[2]), p[3]))
            except ValueError:
                pass                                    # torn last line
    if not nums and not texts:
        return None, 0
    cues = []
    for i, (ms, val) in enumerate(nums):
        end = nums[i + 1][0] if i + 1 < len(nums) else (end_ms or ms + 50)
        cues.append((ms, max(end, ms + 1), val))
    cues += [(ms, max(end, ms + 1), tx) for ms, end, tx in texts]
    cues.sort(key=lambda c: c[0])
    srt = raw_path[: -len(".raw.tsv")] + ".srt"
    with open(srt, "w", encoding="utf-8") as out:
        for i, (a, b, tx) in enumerate(cues, 1):
            out.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{tx}\n\n")
    return srt, len(cues)


class Session:
    """Streams to a raw log (survives a power cut); builds the .srt on stop."""

    def __init__(self, outdir):
        self.outdir, self.active, self.f = outdir, False, None

    def recover(self):
        done = []
        if os.path.isdir(self.outdir):
            for name in sorted(os.listdir(self.outdir)):
                if name.endswith(".raw.tsv"):
                    raw = os.path.join(self.outdir, name)
                    path, _ = raw_to_srt(raw)
                    os.remove(raw)
                    if path:
                        done.append(path)
        return done

    def start(self, t0):
        os.makedirs(self.outdir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_path = os.path.join(self.outdir, stamp + ".raw.tsv")
        self.f = open(self.raw_path, "w", encoding="utf-8")
        self.t0, self.last_t, self.last_flush = t0, t0, time.monotonic()
        self.active = True

    def _ms(self, t):
        return max(round((t - self.t0) * 1000), 0)

    def _flush(self):
        if time.monotonic() - self.last_flush > 1.0:
            self.f.flush()
            self.last_flush = time.monotonic()

    def add_sample(self, t, g):
        if self.active:
            self.last_t = t
            self.f.write(f"S\t{self._ms(t)}\t{'OPEN' if g is None else f'{g:.4f}'}\n")
            self._flush()

    def add_cue(self, t, text, end=None):
        if self.active:
            end = t + 0.5 if end is None else max(end, t + 0.05)
            text = " ".join(text.replace("\t", " ").split())
            self.f.write(f"T\t{self._ms(t)}\t{self._ms(end)}\t{text}\n")
            self._flush()

    @property
    def elapsed(self):
        return self.last_t - self.t0 if self.active else 0.0

    def stop(self):
        if not self.active:
            return None, 0
        self.active = False
        self.f.close()
        path, n = raw_to_srt(self.raw_path, self._ms(self.last_t) + 50)
        os.remove(self.raw_path)
        return path, n


# --- curses UI --------------------------------------------------------------------------
LABEL_W = 6
MIN_W, MIN_H = 50, 22
STRIP_S = 60.0
BLOCKS = "▁▂▃▄▅▆▇█"
GLOW = "·░▒▓█"


def safe_add(win, y, x, s, attr=0):
    try:
        win.addstr(y, x, s, attr)
    except curses.error:
        pass


class App:
    def __init__(self, params, outdir, show_rebounds=True):
        self.det = Detector(params)
        self.sess = Session(outdir)
        self.events = []
        self.strip = deque()                 # (t, g|None), last STRIP_S seconds
        self.g = None
        self.t_origin = self.last_t = None
        self.follow, self.arm = True, 0.0    # tone arm: follow slow baseline, or fixed ln value
        self.range_pct = 30.0
        self.show_rebounds = show_rebounds
        self.note_buf = self.note_t = None
        self.msg, self.msg_until, self.error = "", 0.0, None
        self.glow, self.head = [], None

    def say(self, msg, secs=8.0):
        self.msg, self.msg_until = msg, time.monotonic() + secs

    def tick(self, q):
        samples = []
        try:
            while True:
                item = q.get_nowait()
                if isinstance(item, str):
                    self.error = item
                else:
                    samples.append(item)
        except queue.Empty:
            pass
        if not samples:
            return
        good = [s for s in samples if s[1] is not None]
        if good:
            t = sum(s[0] for s in good) / len(good)
            g = sum(s[1] for s in good) / len(good)
        else:
            t, g = samples[-1][0], None
        if self.t_origin is None:
            self.t_origin = t
        self.last_t, self.g = t, g
        ev = self.det.feed(t, g)
        self.sess.add_sample(t, g)
        self.strip.append((t, g))
        while self.strip and t - self.strip[0][0] > STRIP_S:
            self.strip.popleft()
        if ev:
            self.events.append(ev)
            self.sess.add_cue(ev.t_start, ev.cue(), end=ev.t_peak)

    # needle ----------------------------------------------------------------------
    def update_needle(self, segs):
        if len(self.glow) != segs:
            self.glow, self.head = [0.0] * segs, None
        self.glow = [b * math.exp(-0.05 / 0.5) for b in self.glow]
        if self.det.xs is None or segs < 3:
            self.head = None
            return
        c = segs // 2
        arm = self.det.slow if self.follow else self.arm
        half = math.log1p(self.range_pct / 100)
        raw = c + round((self.det.xs - arm) / half * c)
        head = max(0, min(segs - 1, raw))
        self.pegged = raw < 0 or raw > segs - 1
        if self.head is not None:
            for i in range(*sorted((self.head, head))[:1], sorted((self.head, head))[1] + 1):
                self.glow[i] = max(self.glow[i], 0.9)
        self.glow[head] = 1.0
        self.head = head


def draw(win, app):
    h, w = win.getmaxyx()
    win.erase()
    if h < MIN_H or w < MIN_W:
        safe_add(win, 0, 0, f"terminal too small (need {MIN_W}x{MIN_H})")
        win.refresh()
        return
    det = app.det
    segs = (w - LABEL_W - 2) // 2
    segs = min(segs, 71)
    segs -= (segs % 2 == 0)
    app.update_needle(segs)

    safe_add(win, 0, 2, "ELRON", curses.A_BOLD)
    state_attr = curses.A_REVERSE if det.state == READ else (curses.A_DIM if det.state == IDLE else 0)
    safe_add(win, 0, 10, f" {det.state} ", state_attr)
    if app.sess.active:
        t = int(app.sess.elapsed)
        safe_add(win, 0, 20, f"● REC {t // 60:02d}:{t % 60:02d}", curses.A_BOLD)

    # needle: left = conductance down, right = conductance up
    c = segs // 2
    safe_add(win, 2, 1, "NOW", curses.A_BOLD)
    for i in range(segs):
        if i == app.head:
            ch, at = "█", curses.A_BOLD
        else:
            lvl = app.glow[i]
            ch = GLOW[3] if lvl > 0.7 else GLOW[2] if lvl > 0.4 else GLOW[1] if lvl > 0.15 else (
                "│" if i == c else "·")
            at = curses.A_DIM if lvl <= 0.15 else 0
        safe_add(win, 2, LABEL_W + 2 * i, ch, at)
        safe_add(win, 3, LABEL_W + 2 * i, " ")
    r = f"{app.range_pct:.0f}%"
    safe_add(win, 4, LABEL_W, f"-{r}", curses.A_DIM)
    safe_add(win, 4, LABEL_W + 2 * c - 1, "0", curses.A_BOLD)
    safe_add(win, 4, LABEL_W + 2 * (segs - 1) - len(r), f"+{r}", curses.A_DIM)

    # strip chart: last 60 s, auto-fit
    n = w - 2 - LABEL_W
    safe_add(win, 6, 1, "60s", curses.A_BOLD)
    if app.strip and n > 4 and app.last_t is not None:
        t1 = app.last_t
        t0 = t1 - STRIP_S
        buckets = [[] for _ in range(n)]
        for t, g in app.strip:
            if g:
                buckets[min(n - 1, max(0, int((t - t0) / STRIP_S * n)))].append(math.log(g))
        vals = [sum(b) / len(b) if b else None for b in buckets]
        have = [v for v in vals if v is not None]
        if have:
            lo, hi = min(have), max(have)
            span = max(hi - lo, math.log1p(0.10))
            mid = (hi + lo) / 2
            lo = mid - span / 2
            for k, v in enumerate(vals):
                if v is not None:
                    idx = min(7, max(0, int((v - lo) / span * 8)))
                    safe_add(win, 6, LABEL_W + k, BLOCKS[idx])
        for ev in app.events:
            if ev.t_peak >= t0 and (app.show_rebounds or not ev.rebound):
                k = min(n - 1, int((ev.t_peak - t0) / STRIP_S * n))
                safe_add(win, 7, LABEL_W + k, GLYPH[ev.direction], curses.A_BOLD)

    # recent events
    safe_add(win, 9, 1, "EVENTS", curses.A_BOLD)
    safe_add(win, 9, 12, "  time      amp      dur     peak", curses.A_DIM)
    shown = [e for e in app.events if app.show_rebounds or not e.rebound][-8:]
    t_org = app.t_origin or 0.0
    for k, ev in enumerate(shown):
        line = (f"{ev.t_start - t_org:8.1f}s {GLYPH[ev.direction]} {ev.amp_pct:+7.1f}% "
                f"{ev.duration:6.1f}s {ev.rate_pct_s:7.1f}%/s" + ("  rebound" if ev.rebound else ""))
        safe_add(win, 10 + k, 6, line[: w - 8])

    # status
    y = h - 3
    if app.note_buf is not None:
        safe_add(win, y, 1, ("mark> " + app.note_buf + "_")[: w - 2], curses.A_BOLD)
    elif app.msg and time.monotonic() < app.msg_until:
        safe_add(win, y, 1, app.msg[: w - 2], curses.A_BOLD)
    elif app.error:
        safe_add(win, y, 1, f"ERROR: {app.error}"[: w - 2], curses.A_BOLD)
    elif app.g is None:
        safe_add(win, y, 1, "no contact / out of range" if app.last_t else "waiting for data...",
                 curses.A_DIM)
    else:
        arm = det.slow if app.follow else app.arm
        dev = (math.exp(det.xs - arm) - 1) * 100
        safe_add(win, y, 1,
                 (f"{app.g:9.3f} µS   {dev:+6.1f}% vs {'slow baseline' if app.follow else 'tone arm'}"
                  f" {math.exp(arm):.3f}   {'OFF DIAL' if getattr(app, 'pegged', False) else ''}")[: w - 2],
                 curses.A_DIM)
    safe_add(win, h - 1, 1,
             "q quit  r rec  m mark  s set  f follow  k/l range  h rebounds"[: w - 2], curses.A_DIM)
    win.refresh()


def toggle_record(app):
    s = app.sess
    if s.active:
        path, n = s.stop()
        app.say(f"saved {path} ({n} cues)" if path else "recording empty, nothing saved", 10)
    else:
        s.start(app.last_t if app.last_t is not None else time.monotonic())
        app.say("recording")


def handle_note_key(app, ch):
    if ch in (10, 13, curses.KEY_ENTER):
        text = app.note_buf.strip() or "mark"
        app.sess.add_cue(app.note_t, f"[{text}] READ", end=app.note_t + 5)
        app.say(f"marked: {text}")
        app.note_buf = None
    elif ch == 27:
        app.note_buf = None
    elif ch in (curses.KEY_BACKSPACE, 127, 8):
        app.note_buf = app.note_buf[:-1]
    elif 32 <= ch < 127 and len(app.note_buf) < 200:
        app.note_buf += chr(ch)


def ui_loop(stdscr, app, q):
    next_tick = time.monotonic()
    while True:
        stdscr.timeout(max(0, int((next_tick - time.monotonic()) * 1000)))
        ch = stdscr.getch()
        if app.note_buf is not None:
            if ch != -1:
                handle_note_key(app, ch)
        elif ch != -1:
            key = chr(ch) if 0 <= ch < 256 else ""
            if key in ("q", "Q"):
                return
            elif key in ("r", "R"):
                toggle_record(app)
            elif key in ("m", "M"):
                if app.sess.active and app.last_t is not None:
                    app.note_buf, app.note_t = "", app.last_t
                else:
                    app.say("press r to record first, then m to mark a read")
            elif key in ("s", "S") and app.det.xs is not None:
                app.arm, app.follow = app.det.xs, False
            elif key in ("f", "F"):
                app.follow = True
            elif key in ("h", "H"):
                app.show_rebounds = not app.show_rebounds
            elif key in ("k", "K") or ch == curses.KEY_DOWN:
                app.range_pct = max(2.0, app.range_pct / 1.25)
            elif key in ("l", "L") or ch == curses.KEY_UP:
                app.range_pct = min(400.0, app.range_pct * 1.25)
        if time.monotonic() >= next_tick:
            app.tick(q)
            next_tick += 0.05
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + 0.05
        draw(stdscr, app)


def ui(stdscr, args, q, params):
    curses.curs_set(0)
    stdscr.keypad(True)
    app = App(params, args.outdir, not args.no_rebounds)
    for path in app.sess.recover():
        app.say(f"recovered unfinished session -> {path}", 10)
    try:
        ui_loop(stdscr, app, q)
    finally:
        app.sess.stop()                           # don't lose a session on quit


# --- main ----------------------------------------------------------------------------------
def build_params(sets):
    p, names = Params(), {f.name for f in fields(Params)}
    for item in sets or []:
        k, _, v = item.partition("=")
        k = k.strip().lower()
        if k not in names:
            raise SystemExit(f"unknown parameter {k!r}; choices: {', '.join(sorted(names))}")
        setattr(p, k, float(v))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="fake signal with reads and squeezes")
    ap.add_argument("--replay", metavar="FILE.srt", help="run the detector over a recording")
    ap.add_argument("--play", nargs="?", const=1.0, type=float, metavar="SPEED",
                    help="with --replay: watch it in the UI at SPEEDx instead of a table")
    ap.add_argument("--flip", action="store_true",
                    help="replay: mirror the signal in log space (for a file with the sign reversed)")
    ap.add_argument("--no-rebounds", action="store_true", help="hide rebound events")
    ap.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a Params value")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=ADDR)
    ap.add_argument("--fsr", type=float, default=4.096, choices=sorted(PGA),
                    help="ADS1115 full-scale volts (1.024 = finer steps, but saturates on dry skin)")
    ap.add_argument("--rref", type=float, default=100_000, help="reference resistor, ohms")
    ap.add_argument("--vcc", type=float, default=3.3, help="measured 3V3 rail")
    ap.add_argument("--outdir", default="sessions")
    args = ap.parse_args()
    params = build_params(args.set)

    if args.selftest:
        return selftest(params)
    if args.replay and not args.play:
        return replay_report(args.replay, params, args.flip, not args.no_rebounds)

    q, stop = queue.Queue(), threading.Event()
    if args.replay:
        samples, _ = load_srt(args.replay, args.flip)
        target, targs = run_play, (samples, args.play)
    elif args.sim:
        target, targs = run_sim, ()
    else:
        target, targs = run_i2c, (args.bus, args.addr, args.fsr, args.rref, args.vcc)
    locale.setlocale(locale.LC_ALL, "")                 # needed for block characters
    threading.Thread(target=reader_thread, daemon=True, args=(q, stop, target, *targs)).start()
    try:
        curses.wrapper(ui, args, q, params)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
