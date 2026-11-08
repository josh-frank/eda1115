#!/usr/bin/env python3
"""
elron.py: the electropsychometry live read-out node.

Two state machines run in tandem on one stream of conductance readings:

  SQUEEZE machine (fast, per-frame)       -> events of kind "squeeze"
      Watches the frame-to-frame change. A sharp, sudden move arms it
      (IDLE -> ARMED -> SQUEEZE). It stays in SQUEEZE until the signal goes
      quiet, then asks: did the level come back to where it started?
      Yes = a can squeeze (emitted). No = a level shift, which this machine
      can't judge, so it is NOT emitted: it's the fall machine's territory.

  FALL machine (slow, windowed)           -> events of kind "fall"
      Frame-to-frame change is useless for a fall: a fall is big but
      gradual, and each 50 ms step is lost in the noise. So this machine
      compares two medians: the last 0.5 s against the 2-4 s that came
      before. A drop that stays beyond threshold arms it (ARMED -> FALLING).
      It keeps following the signal down and resolves once it has stopped
      finding new lows for a few seconds. It reports the size, the speed
      and how much of the fall came back (the "flutter").

Arbitration: both machines see most big moves. If a squeeze was just
emitted, a fall overlapping it is vetoed, so one physical event gives one
call. Squeezes win ties because they are the one thing the squeeze machine
is good at.

Both machines share one slow baseline (20 s EMA, frozen while an event is
in progress), exposed as `.baseline`. It doubles as a tracking tone arm.

Direction: internally an "event" is a DROP in the working signal x.
`sign` maps your readings onto x:
    sign=+1  events are RISES in the value you feed in. Right for the
             pimeter's conductance: squeezing the cans lowers resistance, so
             conductance goes up.
    sign=-1  events are DROPS in the value. Right for old Arduino-era files
             like uncle_vic.srt, where the logged value falls on a squeeze.
All numbers in events are positive magnitudes; levels are in your units.

Units: thresholds are adaptive (multiples of measured noise) except the
absolute floors, which are written in uS. Pass scale=1000 for nS.

Hook-up in emeter.py (after em.tick() produces a reading g):
    from elron import Elron
    elron = Elron(scale=K)                     # K = UNIT_SCALE / 1e6
    ...
    for ev in elron.update(g, time.monotonic()):   # g may be None if OPEN
        ...                                        # show / log / mark it

Offline replay of a recorded session, to tune against your annotations:
    python3 elron.py session.srt [--sign -1] [--scale 1]
"""
import math
import re
import statistics
import sys
from collections import deque

IDLE, ARMED, SQUEEZE, FALLING = "IDLE", "ARMED", "SQUEEZE", "FALLING"


class Elron:
    # --- shared -------------------------------------------------------------
    CALIBRATION_FRAMES = 100        # ~5 s at 20 Hz before either machine arms
    BASELINE_TAU_S = 20.0           # slow tonic tracker: "where is home"
    HISTORY_S = 12.0                # raw history kept for the fall machine

    # --- squeeze machine ----------------------------------------------------
    NOISE_TAU_S = 5.0
    NOISE_FLOOR = 1e-4              # uS
    ARM_THRESH_K = 4.0              # frame step must beat K x typical jitter
    DEBOUNCE_FRAMES = 3             # filters single-frame crackle pops
    SETTLE_THRESH_K = 2.0
    SETTLE_FRAMES = 6
    SQUEEZE_TIMEOUT_S = 20.0
    SUSTAIN_TOL_FRAC = 0.30
    SUSTAIN_TOL_K = 3.0

    # --- fall machine -------------------------------------------------------
    CUR_WINDOW_S = 0.5              # "now" = median of the last 0.5 s
    REF_FAR_S, REF_NEAR_S = 4.0, 2.0  # "before" = median from 4 s to 2 s ago
    FALL_K = 4.0                    # drop must beat K x typical window drift
    FALL_HOLD_FRAMES = 6            # ...and hold for 0.3 s
    FALL_NOISE_TAU_S = 30.0         # noise estimate rises slowly...
    FALL_NOISE_DOWN_S = 5.0         # ...but falls fast, so it tracks the quiet level
    FALL_NOISE_FLOOR = 2e-3         # uS
    FALL_QUIET_FRAC = 0.25          # "quiet" = drop below this x threshold
    FALL_MIN_AMP = 0.3              # uS; smaller than this is never a fall
    FALL_SETTLE_S = 3.0             # no new low for this long = fall is over
    FALL_NEWLOW_K = 1.0             # a "new low" must beat the old by K x noise
    FALL_TIMEOUT_S = 90.0

    # --- arbitration --------------------------------------------------------
    VETO_BEFORE_S, VETO_AFTER_S = 1.0, 3.0

    def __init__(self, scale=1.0, sign=+1, **tuning):
        self.sign = +1 if sign >= 0 else -1
        self.noise_floor = self.NOISE_FLOOR * scale
        self.fall_floor = self.FALL_NOISE_FLOOR * scale
        self.fall_min_amp = self.FALL_MIN_AMP * scale
        for k, v in tuning.items():
            if not hasattr(self, k):
                raise TypeError(f"unknown tuning parameter {k!r}")
            setattr(self, k, v)
        self.reset()

    # -- public -----------------------------------------------------------------
    @property
    def state(self):
        return f"squeeze:{self.sq_state} fall:{self.fl_state}"

    def reset(self):
        """Forget everything (also what a fresh connect should do)."""
        self.baseline = None            # in the user's units
        self._xb = None                 # baseline in working units
        self._t = self._x = None
        self._hist = deque()
        self._squeezes = []             # recent emitted squeezes (t0, t1)
        self._sq_reset()
        self._fl_reset()

    def update(self, value, t):
        """Feed one reading (value in your units, t in seconds).
        value=None means out of range / no contact. Returns a list of event
        dicts that completed on this reading (usually empty)."""
        if value is None:
            self._abort()
            return []
        x = -self.sign * value
        if self._t is not None and t - self._t > 1.0:
            self._abort()               # a hole in the data: start fresh
        dt = 0.05 if self._t is None else min(max(t - self._t, 0.001), 1.0)
        delta = 0.0 if self._x is None else x - self._x
        self._t, self._x = t, x
        if self._xb is None:
            self._xb = x

        self._hist.append((t, x))
        while self._hist and t - self._hist[0][0] > self.HISTORY_S:
            self._hist.popleft()

        sq = self._squeeze_tick(x, delta, dt, t)
        fl = self._fall_tick(dt, t)
        self.baseline = self._u(self._xb)

        events = []
        if sq and sq["kind"] == "squeeze":      # level shifts are swallowed
            self._squeezes.append((sq["t"], sq["t_end"]))
            events.append(sq)
        self._squeezes = [s for s in self._squeezes if t - s[1] < 120]
        if fl and not self._vetoed(fl):
            events.append(fl)
        return events

    # -- helpers ----------------------------------------------------------------
    def _u(self, x):
        """Working units back to the user's units."""
        return -self.sign * x

    def _median(self, t0, t1):
        vals = [x for (tt, x) in self._hist if t0 <= tt <= t1]
        return statistics.median(vals) if len(vals) >= 3 else None

    def _vetoed(self, fl):
        return any(fl["t"] <= s1 + self.VETO_AFTER_S and
                   fl["t_low"] + 1.0 >= s0 - self.VETO_BEFORE_S
                   for s0, s1 in self._squeezes)

    def _abort(self):
        """Lost contact / data hole: drop any event in progress, keep going."""
        self._t = self._x = None
        self._hist.clear()
        if self.sq_state in (SQUEEZE,):
            self.sq_state = ARMED
        self.sq_deb = self.sq_settle = 0
        if self.fl_state == FALLING:
            self.fl_state = ARMED
        self.fl_hold = 0
        self.fl_cool = None

    def _baseline_free(self):
        return self.sq_state != SQUEEZE and self.fl_state != FALLING

    # -- squeeze machine: IDLE -> ARMED -> SQUEEZE -> ARMED ---------------------------
    def _sq_reset(self):
        self.sq_state = IDLE
        self.sq_noise = None
        self.sq_cal = 0
        self.sq_deb = 0
        self.sq_deb_t0 = None
        self.sq_settle = 0
        self.sq_prev_sign = 0
        self.sq_run = 0
        self.sq_pre = self.sq_t0 = self.sq_peak = self.sq_peak_t = None

    def _squeeze_tick(self, x, delta, dt, t):
        a_noise = min(dt / self.NOISE_TAU_S, 1)
        a_base = min(dt / self.BASELINE_TAU_S, 1)

        if self.sq_state == IDLE:
            if self.sq_noise is None:
                self.sq_noise = max(abs(delta), self.noise_floor)
            else:
                if self._baseline_free():
                    self._xb += (x - self._xb) * a_base
                self.sq_noise += (abs(delta) - self.sq_noise) * a_noise
                self.sq_noise = max(self.sq_noise, self.noise_floor)
            self.sq_cal += 1
            if self.sq_cal >= self.CALIBRATION_FRAMES:
                self.sq_state = ARMED
            return None

        if self.sq_state == ARMED:
            thresh = self.ARM_THRESH_K * self.sq_noise
            if delta < -thresh:
                if self.sq_deb == 0:
                    self.sq_deb_t0 = t
                self.sq_deb += 1
            else:
                self.sq_deb = 0
                if self._baseline_free():
                    self._xb += (x - self._xb) * a_base
                # feed the noise estimate only back-and-forth jitter, not a
                # sustained run one way, and cap each step so one crackle pop
                # can't inflate the threshold meant to ignore it
                s = (delta > 0) - (delta < 0)
                if s != 0 and s == self.sq_prev_sign:
                    self.sq_run += 1
                else:
                    self.sq_run, self.sq_prev_sign = 1, s
                if self.sq_run <= 2:
                    step = min(abs(delta), 3 * self.sq_noise)
                    self.sq_noise += (step - self.sq_noise) * a_noise
                    self.sq_noise = max(self.sq_noise, self.noise_floor)
            if self.sq_deb >= self.DEBOUNCE_FRAMES:
                self.sq_pre = self._xb
                self.sq_t0 = self.sq_deb_t0
                self.sq_peak, self.sq_peak_t = x, t
                self.sq_settle = self.sq_deb = 0
                self.sq_state = SQUEEZE
            return None

        # SQUEEZE
        if x < self.sq_peak:
            self.sq_peak, self.sq_peak_t = x, t
        settle = self.SETTLE_THRESH_K * self.sq_noise
        self.sq_settle = self.sq_settle + 1 if abs(delta) < settle else 0
        if self.sq_settle >= self.SETTLE_FRAMES or t - self.sq_t0 >= self.SQUEEZE_TIMEOUT_S:
            return self._squeeze_resolve(x, t)
        return None

    def _squeeze_resolve(self, x, t):
        amp = self.sq_pre - self.sq_peak
        tol = max(self.SUSTAIN_TOL_K * self.sq_noise,
                  self.SUSTAIN_TOL_FRAC * abs(amp))
        offset = abs(x - self.sq_pre)
        returned = offset <= tol
        margin = (tol - offset) / tol if tol > 0 else 0.0
        conf = min(max(0.5 + 0.5 * (margin if returned else -margin), 0.5), 0.99)
        ev = {
            "kind": "squeeze" if returned else "shift",
            "t": self.sq_t0, "t_end": t,
            "amplitude": amp,
            "attack_s": self.sq_peak_t - self.sq_t0,
            "release_s": t - self.sq_peak_t,
            "baseline": self._u(self.sq_pre),
            "sustain": self._u(x),
            "confidence": conf,
        }
        self._xb = x                    # re-center, wherever it landed
        self.sq_state = ARMED
        self.sq_pre = None
        return ev

    # -- fall machine: IDLE -> ARMED -> FALLING -> (cooldown) -> ARMED -----------------
    def _fl_reset(self):
        self.fl_state = IDLE
        self.fl_noise = None
        self.fl_cal = 0
        self.fl_hold = 0
        self.fl_cool = None             # no re-arming until this time
        self.fl_pre = self.fl_t0 = None
        self.fl_low = self.fl_low_t = self.fl_mark = self.fl_mark_t = None
        self.fl_thr = None

    def _fall_tick(self, dt, t):
        cur = self._median(t - self.CUR_WINDOW_S, t)
        ref = self._median(t - self.REF_FAR_S, t - self.REF_NEAR_S)
        if cur is None or ref is None:
            return None                 # not enough history yet
        drop = ref - cur                # positive = the signal fell
        if self.fl_state == IDLE:
            if self.fl_noise is None:
                self.fl_noise = max(abs(drop), self.fall_floor)
            else:
                self._fall_noise(abs(drop), dt)
            self.fl_cal += 1
            if self.fl_cal >= self.CALIBRATION_FRAMES:
                self.fl_state = ARMED
            return None

        if self.fl_state == ARMED:
            thr = self.FALL_K * self.fl_noise
            cooling = self.fl_cool is not None and t < self.fl_cool
            # winsorised: a fall in progress can't inflate its own threshold
            self._fall_noise(min(abs(drop), thr), dt)
            self.fl_hold = self.fl_hold + 1 if (drop > thr and not cooling) else 0
            if self.fl_hold >= self.FALL_HOLD_FRAMES:
                onset = self._find_onset(ref, thr, t)
                pre = self._median(onset - 2.0, onset)
                self.fl_pre = pre if pre is not None else ref
                self.fl_t0 = onset
                self.fl_low = self.fl_mark = cur
                self.fl_low_t = self.fl_mark_t = t
                self.fl_thr = thr
                self.fl_hold = 0
                self.fl_state = FALLING
            return None

        # FALLING: follow the signal down until it stops finding new lows
        if cur < self.fl_low:
            self.fl_low, self.fl_low_t = cur, t
        if cur < self.fl_mark - self.FALL_NEWLOW_K * self.fl_noise:
            self.fl_mark, self.fl_mark_t = cur, t
        if (t - self.fl_mark_t >= self.FALL_SETTLE_S
                or t - self.fl_t0 >= self.FALL_TIMEOUT_S):
            return self._fall_resolve(cur, t)
        return None

    def _find_onset(self, ref, thr, t):
        """When did the descent start? Walk back from now to the last moment
        the signal (5-sample median, so crackles don't fool it) was still
        within a quarter-threshold of its earlier level. Slow drift can fool
        a 'last quiet moment' test; this can't."""
        h = [x for (_, x) in self._hist]
        ts = [tt for (tt, _) in self._hist]
        for i in range(len(h) - 3, 1, -1):
            if statistics.median(h[i - 2:i + 3]) >= ref - self.FALL_QUIET_FRAC * thr:
                return ts[i]
        return ts[0]

    def _fall_noise(self, size, dt):
        """Asymmetric EMA of the window drift: slow up, fast down. The start-up
        settle and any big fall can't leave the threshold inflated for long."""
        tau = self.FALL_NOISE_TAU_S if size > self.fl_noise else self.FALL_NOISE_DOWN_S
        self.fl_noise += (size - self.fl_noise) * min(dt / tau, 1)
        self.fl_noise = max(self.fl_noise, self.fall_floor)

    def _fall_resolve(self, cur, t):
        amp = self.fl_pre - self.fl_low
        ev = None
        if amp >= self.fall_min_amp:
            ev = {
                "kind": "fall",
                "t": self.fl_t0, "t_end": t, "t_low": self.fl_low_t,
                "amplitude": amp,
                "attack_s": self.fl_low_t - self.fl_t0,
                "duration_s": t - self.fl_t0,
                "recovery": min(max((cur - self.fl_low) / amp, 0.0), 1.0),
                "strength": amp / max(self.fl_noise, 1e-12),
                "baseline": self._u(self.fl_pre),
                "low": self._u(self.fl_low),
                "sustain": self._u(cur),
            }
        self.fl_state = ARMED
        self.fl_hold = 0
        # after a real fall the "before" window still holds it for a few
        # seconds, so don't re-trigger on its own tail. A candidate too small
        # to report gets only a brief pause, so it can't mask a real fall
        # that starts right behind it.
        self.fl_cool = t + (self.REF_FAR_S + 0.5 if ev else 1.0)
        return ev


# --- offline replay -----------------------------------------------------------
def _tc(s):
    h, m, rest = s.strip().split(":")
    sec, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(sec)) + int(ms) / 1000


def load_srt(path):
    """-> (frames [(t_s, value|None)], notes [(t0, t1, text, read)])"""
    frames, notes = [], []
    for block in open(path, encoding="utf-8").read().strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        a, b = (_tc(x) for x in lines[1].split("-->"))
        body = "\n".join(lines[2:]).strip()
        m = re.match(r"^\[(.*)\](\s*READ)?$", body, re.S)
        if m:
            notes.append((a, b, m.group(1), bool(m.group(2))))
        elif body == "OPEN":
            frames.append((a, None))
        else:
            try:
                frames.append((a, float(body)))
            except ValueError:
                pass
    return frames, notes


def replay(path, sign, scale):
    frames, notes = load_srt(path)
    el = Elron(scale=scale, sign=sign)
    events = []
    for t, v in frames:
        events += el.update(v, t)
    mmss = lambda s: f"{int(s // 60):02d}:{s % 60:04.1f}"
    print(f"{len(frames)} frames, {len(events)} events, sign={sign:+d}, scale={scale:g}")
    for ev in events:
        extra = (f"recovery {ev['recovery']:.0%}  strength {ev['strength']:.1f}"
                 if ev["kind"] == "fall" else f"conf {ev['confidence']:.2f}")
        print(f"  {mmss(ev['t'])}  {ev['kind']:8s} amp {ev['amplitude']:.3f}  "
              f"attack {ev['attack_s']:.1f}s  {extra}")
    if notes:
        hit = 0
        for a, b, text, _ in notes:
            got = [e for e in events if a - 3 <= e["t"] <= b]
            hit += bool(got)
            print(f"  note {mmss(a)}-{mmss(b)}: " +
                  (", ".join(e["kind"] for e in got) or "MISSED"))
        stray = [e for e in events if not any(a - 3 <= e["t"] <= b for a, b, *_ in notes)]
        print(f"  {hit}/{len(notes)} notes hit, {len(stray)} events outside any note")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="replay an .srt session through Elron")
    ap.add_argument("srt")
    ap.add_argument("--sign", type=int, default=+1,
                    help="+1: events are rises (pimeter conductance), -1: drops (old Arduino files)")
    ap.add_argument("--scale", type=float, default=1.0, help="units per uS (1000 for nS)")
    args = ap.parse_args()
    replay(args.srt, args.sign, args.scale)
