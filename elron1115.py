#!/usr/bin/env python3
"""
elron1115.py: electrodermography read detection state machine.
No GUI, no I/O, no threads: feed it samples, it hands back events.

    IDLE   no contact, or the signal is still settling. No events are emitted.
    ARM    watching. Any change bigger and faster than the thresholds opens a READ.
    READ   one monotone excursion is open. It closes when the signal gives back
           half of it, stops making progress, or times out. Each READ becomes
           one Event.

It only measures: direction (+1 conductance up / -1 conductance down), amplitude
in %, time to peak, and peak speed in %/s. What an event MEANS is up to you.
Everything works in log-conductance, so thresholds are percent changes and the
same settings serve a dry day and a sweaty one.

An excursion ends at its extreme and the next can start right there, so a held
squeeze is two events (rise, then a fast opposite event starting at the peak).
Events that start at the previous event's peak, in the opposite direction, are
tagged `rebound`: structure only, filter them if you don't want them.

Use:
    from elron1115 import Detector, load_params, srt_header

    det = Detector(load_params())            # ./default.elron1115.toml, else built-in defaults
    det = Detector(load_params("cans"))      # ./cans.elron1115.toml (or a path to any .toml)
    det = Detector(load_params(sets=["start_pct=8"]))   # --set style overrides on top
    ...
    ev = det.feed(t, g)                      # t: seconds (any origin), g: uS or None
    det.state                                # "IDLE" | "ARM" | "READ"   (after each feed)
    if ev:                                   # an Event, or None
        ev.as_dict()                         # JSON-ready
    ...
    ev = det.flush()                         # end of data: closes an open READ

Plugging into stream.py (publish() already gets every sample):
    frame["s"] = det.state                   # state rides along on each frame
    ev = det.feed(frame["t"], frame["g"])
    if ev: announce_event(ev.as_dict())      # e.g. an SSE "ev" event; NOT written to recordings

Notes:
  * g=None (no contact) or a gap in t longer than Params.gap_s sends it to IDLE and
    throws away any open READ (counted in det.aborted).
  * Events appear when a READ CLOSES, i.e. after the fact. det.state == "READ" is
    how you know one is open right now.
  * Timings were tuned at ~20-50 Hz. The smoothing and rate windows are in seconds
    and work at any rate, but a 0.5 s rate window is only a few samples at 10 Hz.
  * Not thread-safe: call feed() from one thread (the reader).
"""
import bisect
import json
import math
import tomllib                      # Python 3.11+; on older: `import tomli as tomllib`
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

IDLE, ARM, READ = "IDLE", "ARM", "READ"


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

    def as_dict(self):
        """JSON-ready: times in the same seconds you fed in, values in your units."""
        return {
            "direction": self.direction,          # +1 conductance up, -1 down
            "t_start": round(self.t_start, 3),
            "t_peak": round(self.t_peak, 3),
            "t_end": round(self.t_end, 3),
            "duration": round(self.duration, 3),  # start -> peak
            "pre_g": round(self.pre_g, 4),
            "peak_g": round(self.peak_g, 4),
            "amp_pct": round(self.amp_pct, 2),
            "rate_pct_s": round(self.rate_pct_s, 2),
            "rebound": self.rebound,
        }


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

    @property
    def slow_g(self):
        """Slow baseline in uS (context only), or None before the first sample."""
        return None if self.slow is None else math.exp(self.slow)

    def reset(self):
        """Forget everything: state, history, slow baseline, abort count."""
        self.slow = None
        self.aborted = 0
        self.ev = None                              # a reset is not an aborted READ
        self._to_idle()

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


# --- profiles: params from TOML, then --set overrides -------------------------
# Precedence, lowest to highest: dataclass defaults < TOML profile < --set overrides.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = BASE_DIR / "default.elron1115.toml"
PROFILE_SUFFIX = ".elron1115.toml"


def resolve_profile(profile=None):
    """-> Path of the TOML file to use, or None for the built-in defaults.

    profile=None   -> default.elron1115.toml next to this file, if it exists, else None.
    "cans"         -> cans.elron1115.toml next to this file.
    anything else  -> treated as a path (absolute, or relative to the working directory).
    An explicitly requested profile that doesn't exist raises FileNotFoundError.
    """
    if profile is None:
        return DEFAULT_PROFILE if DEFAULT_PROFILE.is_file() else None
    prof = str(profile)
    if "/" not in prof and "\\" not in prof and not prof.endswith(".toml"):
        path = BASE_DIR / (prof + PROFILE_SUFFIX)
    else:
        path = Path(prof).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"profile {profile!r} not found (looked for {path})")
    return path


def params_from_toml(path, base=None):
    """Flat TOML of overrides -> Params, applied on top of `base` (or the defaults).
    Raises ValueError on an unknown name or a non-numeric value."""
    with open(path, "rb") as fh:                    # tomllib wants binary mode
        data = tomllib.load(fh)
    p, names = base or Params(), {f.name for f in fields(Params)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{path}: unknown parameter(s) {unknown}; choices: {', '.join(sorted(names))}")
    for k, v in data.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{path}: {k} must be a number, got {v!r}")
        setattr(p, k, float(v))
    return p


def params_from_sets(sets, base=None):
    """["start_pct=8", "min_rate_pct_s=2"] -> Params, on top of `base` (or the defaults).
    Raises ValueError on a bad name."""
    p, names = base or Params(), {f.name for f in fields(Params)}
    for item in sets or []:
        k, _, v = item.partition("=")
        k = k.strip().lower()
        if k not in names:
            raise ValueError(f"unknown parameter {k!r}; choices: {', '.join(sorted(names))}")
        setattr(p, k, float(v))
    return p


def load_params(profile=None, sets=None):
    """The whole stack: defaults < TOML profile < --set overrides. -> Params."""
    base = Params()
    path = resolve_profile(profile)
    if path is not None:
        base = params_from_toml(path, base)
    return params_from_sets(sets, base)


def add_profile_args(parser):
    """Add --profile and --set to an argparse parser; pair with params_from_args()."""
    parser.add_argument("--profile", metavar="NAME_OR_PATH",
                        help="TOML profile: a name (NAME.elron1115.toml next to elron1115.py) "
                             "or a path. Default: default.elron1115.toml")
    parser.add_argument("--set", dest="sets", action="append", metavar="NAME=VALUE",
                        help="override one parameter on top of the profile (repeatable)")


def params_from_args(args):
    return load_params(getattr(args, "profile", None), getattr(args, "sets", None))


# --- recording header: params as an SRT cue at a negative timestamp ------------
# SRT has no comment syntax, so the params ride in cue 0, placed before t=0. Negative
# timestamps aren't valid SRT: lenient parsers (and a plain text reader) cope, strict
# players may skip or choke on the cue. Tune the two constants if one does.
SRT_HEADER_START_S = -2.0
SRT_HEADER_END_S = -1.0


def _srt_ts(sec):
    sign = "-" if sec < 0 else ""
    ms = round(abs(sec) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{sign}{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def profile_label(profile=None):
    """Human label for where params came from: the profile filename, or 'built-in defaults'."""
    path = resolve_profile(profile)
    return path.name if path else "built-in defaults"


def srt_header(params, profile=None, label=None):
    """The effective params as one SRT cue (index 0, negative timestamps), ending in a
    blank line. Write it first when a recording starts, before any real cue.

    `profile` is whatever was passed to load_params (None = default); `label` overrides
    the profile line outright (used when rebuilding a header from a crashed session's log,
    where the profile file may have changed or gone). The body is valid TOML, so it can be
    read back with tomllib after dropping the cue's first two lines.
    """
    lines = [
        "# elron1115 params",
        f"profile = {json.dumps(label if label is not None else profile_label(profile))}",
    ]
    lines += [f"{f.name} = {getattr(params, f.name)!r}" for f in fields(Params)]
    return (f"0\n{_srt_ts(SRT_HEADER_START_S)} --> {_srt_ts(SRT_HEADER_END_S)}\n"
            + "\n".join(lines) + "\n\n")