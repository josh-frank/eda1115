#!/usr/bin/env python3
"""
emeter.py: an e-meter style display for your EDA rig, in curses.

The "needle" is a row of LED segments like a mixer meter, with a glowing
head that sweeps and fades. Two knobs sit above it:

    TONE ARM (scroll)       A / D   -/+   which conductance sits at dial center
    SENSITIVITY (zoom)      K / L   -/+   how far the needle swings per uS

Hold Shift (A D K L) for coarse steps. Arrow keys also work:
left/right = tone arm, down/up = sensitivity.

Other keys:
    s   SET: drop the tone arm onto the current reading (needle to center)
    b   switch between glowing-needle and fill-bar display
    q   quit

Top meter is the live needle, bottom meter is a ~2 s average. The scale
between them is in µS (or nS: see UNIT_SCALE) relative to the tone arm setting
and auto-ranges with
sensitivity. Data comes from an ADS1115 via smbus2, or --sim for a fake signal.
"""
import argparse
import curses
import locale
import math
import queue
import random
import threading
import time

# --- hardware -------------------------------------------------------------
ADDR = 0x48
REG_CONV, REG_CONFIG = 0x00, 0x01
# AIN0 vs GND, +/-4.096 V, continuous, 128 SPS, comparator off
CONFIG = 0x4000 | 0x0200 | 0x0000 | 0x0080 | 0x0003
LSB = 4.096 / 32768
R_REF, VCC = 100_000, 3.3

# ---- units (edit this) ----
UNIT_SCALE = 1e9           # 1e6 = µS (default), 1e9 = nS
UNIT = {1e6: "µS", 1e9: "nS"}[UNIT_SCALE]
K = UNIT_SCALE / 1e6       # the range constants below are written in µS, then scaled by K


def fmt(x):
    """Compact number for readouts, whatever the unit."""
    a = abs(x)
    return (f"{x:.3f}" if a < 10 else f"{x:.2f}" if a < 100 else
            f"{x:.1f}" if a < 1000 else f"{x:.0f}")


# --- behaviour --------------------------------------------------------------
TICK_MS = 50               # 20 updates a second
TRAIL_TAU_S = 0.5          # how long the needle glow takes to fade
AVG_TAU_S = 2.0            # smoothing for the bottom meter
# ---- knob ranges (edit these; written in µS, scaled by K for your unit) ----
TA_MIN, TA_MAX = -30.0 * K, 30.0 * K      # tone arm range (dial center = this value)
HALF_MIN, HALF_MAX = 0.02 * K, 20.0 * K   # dial half-width range (sensitivity knob)
START_HALF = 0.25 * K                     # starting dial half-width
COARSE = 5                 # Shift multiplier

# --- layout -----------------------------------------------------------------
LABEL_W = 6                # left margin for "NOW" / "AVG"
SEG = "▐█▌"                # one LED: 2 cells wide, half-cell gaps either side
SEG_W = 3
MIN_W, MIN_H = 40, 18
Y_KNOB_LABEL, Y_KNOB, Y_KNOB_VAL, Y_KNOB_KEYS = 2, 3, 6, 7
Y_NOW, Y_TICKS, Y_LABELS, Y_AVG = 9, 11, 12, 13
Y_STATUS, Y_HELP = 16, 17

KNOB_RING = ["╭─────╮",
             "│     │",
             "╰     ╯"]
# clockwise from 7 o'clock to 5 o'clock, the gap sits at the bottom
KNOB_PATH = [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3),
             (0, 4), (0, 5), (0, 6), (1, 6), (2, 6)]


# --- pure helpers (no curses needed) ---------------------------------------
def gradient(f):
    """0..1 -> (r, g, b): green / amber / red."""
    stops = [(0.0, (0, 220, 90)), (0.5, (255, 200, 0)), (1.0, (255, 50, 40))]
    f = max(0.0, min(1.0, f))
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f <= f1:
            t = (f - f0) / (f1 - f0)
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return stops[-1][1]


def rgb_to_256(rgb):
    r, g, b = (round(c / 255 * 5) for c in rgb)
    return 16 + 36 * r + 6 * g + b


def scaled(rgb, k):
    return tuple(int(c * k) for c in rgb)


def nice_step(min_step):
    """Smallest 1/2/5 x 10^n that is >= min_step."""
    exp = math.floor(math.log10(min_step))
    for mult in (1, 2, 5, 10):
        step = mult * 10 ** exp
        if step >= min_step * (1 - 1e-9):
            return step


def dial_segments(width):
    """Odd number of LED segments that fit the terminal, with a center one."""
    n = min((width - LABEL_W - 2) // SEG_W, 61)
    return n if n % 2 else n - 1


def head_index(dev, half, segs):
    """Needle segment for a deviation (uS) -> (index, pegged off the dial)."""
    c = segs // 2
    raw = c + round(dev / half * c)
    return max(0, min(segs - 1, raw)), raw < 0 or raw > segs - 1


def dial_ticks(half, segs):
    """[(segment index, label)] on round uS values, never crowding."""
    c = segs // 2
    seg_us = half / c
    step = nice_step(3.5 * seg_us)
    dec = max(0, -math.floor(math.log10(step) + 1e-9))
    ticks = [(c, "0")]
    n = 1
    while True:
        off = round(n * step / seg_us)
        if off > c:
            break
        v = n * step
        ticks.append((c + off, f"{v:+.{dec}f}"))
        ticks.append((c - off, f"{-v:+.{dec}f}"))
        n += 1
    return ticks


def knob_pointer(t):
    """Knob position 0..1 -> (row, col) on the ring."""
    t = max(0.0, min(1.0, t))
    return KNOB_PATH[round(t * (len(KNOB_PATH) - 1))]


def seg_levels(needle, segs, mode):
    """Brightness level 0..3 for every segment."""
    lv = [0] * segs
    if mode == "needle":
        for i, b in enumerate(needle.br):
            lv[i] = 3 if b > 0.75 else 2 if b > 0.45 else 1 if b > 0.15 else 0
    elif needle.head is not None:
        lo, hi = sorted((segs // 2, needle.head))
        for i in range(lo, hi + 1):
            lv[i] = 3
    return lv


# --- data sources (background thread) ---------------------------------------
def run_i2c(q, stop, bus_no, addr):
    from smbus2 import SMBus
    with SMBus(bus_no) as bus:
        bus.write_i2c_block_data(addr, REG_CONFIG, [CONFIG >> 8, CONFIG & 0xFF])
        time.sleep(0.02)
        while not stop.is_set():
            hi, lo = bus.read_i2c_block_data(addr, REG_CONV, 2)
            v = int.from_bytes([hi, lo], "big", signed=True) * LSB
            if 0.001 < v < VCC - 0.001:
                q.put((VCC - v) / (v * R_REF) * UNIT_SCALE)
            else:
                q.put(None)                              # out of range
            time.sleep(0.02)


def run_sim(q, stop):
    """Fake EDA: slow wander + occasional skin-conductance responses + noise."""
    t0 = time.time()
    bumps, next_bump = [], 3.0
    while not stop.is_set():
        t = time.time() - t0
        if t >= next_bump:
            bumps.append((t, random.uniform(0.05, 0.4)))
            next_bump = t + random.uniform(4, 12)
        g = 2.5 + 0.08 * math.sin(t / 37) + 0.02 * math.sin(t / 5.3)
        for t_b, amp in bumps:
            dt = t - t_b
            g += amp * (1 - math.exp(-dt / 0.4)) * math.exp(-dt / 4.0)
        bumps = [(tb, a) for tb, a in bumps if t - tb < 40]
        q.put((g + random.gauss(0, 0.008)) * K)
        time.sleep(0.02)


def reader_thread(q, stop, sim, bus_no, addr):
    try:
        run_sim(q, stop) if sim else run_i2c(q, stop, bus_no, addr)
    except Exception as e:                       # surface errors in the UI
        q.put(f"{type(e).__name__}: {e}")


# --- state ------------------------------------------------------------------
class Needle:
    """Per-segment glow that decays; the head is the live position."""

    def __init__(self):
        self.br, self.head, self.pegged = [], None, False

    def resize(self, n):
        self.br, self.head = [0.0] * n, None

    def update(self, head, decay):
        self.br = [b * decay for b in self.br]
        if self.head is not None:                       # light the sweep path
            lo, hi = sorted((self.head, head))
            for i in range(lo, hi + 1):
                self.br[i] = max(self.br[i], 0.9)
        self.br[head] = 1.0
        self.head = head


class EMeter:
    def __init__(self, q):
        self.q = q
        self.segs = 0
        self.now, self.slow = Needle(), Needle()
        self.g = self.avg = self.ta = None
        self.half = START_HALF
        self.mode = "needle"
        self.contact = True
        self.error = None
        self.avg_alpha = 1 - math.exp(-(TICK_MS / 1000) / AVG_TAU_S)
        self.decay = math.exp(-(TICK_MS / 1000) / TRAIL_TAU_S)

    # knobs ------------------------------------------------------------------
    @property
    def seg_us(self):
        return self.half / max(1, self.segs // 2)

    def turn_ta(self, n):
        """n > 0 raises the tone arm. One step = half a LED at current zoom."""
        if self.ta is not None:
            self.ta = max(TA_MIN, min(TA_MAX, self.ta + n * 0.5 * self.seg_us))

    def turn_sens(self, n):
        """n > 0 = more sensitive (needle swings further)."""
        self.half = max(HALF_MIN, min(HALF_MAX, self.half * 1.1 ** (-n)))

    def set_needle(self):
        if self.g is not None:
            self.ta = self.g

    @property
    def ta_pos(self):
        return 0.0 if self.ta is None else (self.ta - TA_MIN) / (TA_MAX - TA_MIN)

    @property
    def sens_pos(self):
        return math.log(HALF_MAX / self.half) / math.log(HALF_MAX / HALF_MIN)

    def set_segments(self, n):
        if n != self.segs:
            self.segs = n
            self.now.resize(n)
            self.slow.resize(n)

    # data ---------------------------------------------------------------------
    def tick(self):
        gs, bad = [], False
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, str):
                    self.error = item
                elif item is None:
                    bad = True
                else:
                    gs.append(item)
        except queue.Empty:
            pass

        if gs:
            self.g = sum(gs) / len(gs)
            self.contact = True
            if self.ta is None:
                self.ta = self.g                      # first reading sets the arm
            self.avg = self.g if self.avg is None else \
                self.avg + (self.g - self.avg) * self.avg_alpha
        elif bad:
            self.contact = False

        if self.g is None or self.ta is None or self.segs < 3:
            return
        for needle, val in ((self.now, self.g), (self.slow, self.avg)):
            head, needle.pegged = head_index(val - self.ta, self.half, self.segs)
            needle.update(head, self.decay)

    def status(self):
        if self.error:
            return f"ERROR: {self.error}"
        if self.g is None:
            return "waiting for data..."
        msg = (f"reading {self.g:6.2f} {UNIT}   tone arm {self.ta:5.2f} {UNIT}   "
               f"dial ±{fmt(self.half)} {UNIT}   {self.mode}")
        if self.now.pegged:
            msg += "   OFF DIAL" + (" >" if self.g > self.ta else " <")
        if not self.contact:
            msg += "   [out of range, check contact]"
        return msg


# --- curses UI --------------------------------------------------------------
class Palette:
    """Hands out curses color pairs on demand, keyed by foreground color."""

    def __init__(self, bg):
        self.bg, self.cache, self.next = bg, {}, 1

    def attr(self, fg, extra=0):
        if fg not in self.cache:
            if self.next < curses.COLOR_PAIRS:
                curses.init_pair(self.next, fg, self.bg)
                self.cache[fg] = curses.color_pair(self.next)
                self.next += 1
            else:
                self.cache[fg] = 0
        return self.cache[fg] | extra


def safe_add(win, y, x, s, attr=0):
    try:
        win.addstr(y, x, s, attr)
    except curses.error:
        pass


def seg_attr(pal, rich, i, c, level, is_head, pegged):
    dist = abs(i - c) / max(c, 1)
    if is_head:
        if rich:
            return pal.attr(196 if pegged else 231, curses.A_BOLD)
        return pal.attr(curses.COLOR_RED if pegged else curses.COLOR_WHITE, curses.A_BOLD)
    if level == 0:
        return pal.attr(236) if rich else pal.attr(curses.COLOR_WHITE, curses.A_DIM)
    if rich:
        k = (0.28, 0.55, 1.0)[level - 1]
        return pal.attr(rgb_to_256(scaled(gradient(dist), k)))
    base = (curses.COLOR_GREEN if dist < 0.4 else
            curses.COLOR_YELLOW if dist < 0.75 else curses.COLOR_RED)
    return pal.attr(base, (curses.A_DIM, 0, curses.A_BOLD)[level - 1])


def draw_meter(win, pal, rich, y, label, needle, segs, mode):
    c = segs // 2
    levels = seg_levels(needle, segs, mode)
    safe_add(win, y, 1, label, curses.A_BOLD)
    for row in (0, 1):
        for i in range(segs):
            attr = seg_attr(pal, rich, i, c, levels[i], i == needle.head, needle.pegged)
            safe_add(win, y + row, LABEL_W + SEG_W * i, SEG, attr)


def draw_scale(win, y_ticks, half, segs, w):
    for idx, label in dial_ticks(half, segs):
        cx = LABEL_W + SEG_W * idx + 1
        safe_add(win, y_ticks, cx, "╵", curses.A_DIM)
        x = max(0, min(w - len(label) - 1, cx - len(label) // 2))
        safe_add(win, y_ticks + 1, x, label, curses.A_BOLD if label == "0" else 0)


def draw_knob(win, pal, rich, x, title, t, value, keys):
    ring = pal.attr(244) if rich else pal.attr(curses.COLOR_WHITE, curses.A_DIM)
    dot = pal.attr(214, curses.A_BOLD) if rich else pal.attr(curses.COLOR_YELLOW, curses.A_BOLD)
    safe_add(win, Y_KNOB_LABEL, x, title, curses.A_BOLD)
    for r, row in enumerate(KNOB_RING):
        safe_add(win, Y_KNOB + r, x, row, ring)
    pr, pc = knob_pointer(t)
    safe_add(win, Y_KNOB + pr, x + pc, "●", dot)
    safe_add(win, Y_KNOB_VAL, x, value)
    safe_add(win, Y_KNOB_KEYS, x, keys, curses.A_DIM)


def draw(win, pal, rich, em):
    h, w = win.getmaxyx()
    win.erase()
    if h < MIN_H or w < MIN_W:
        safe_add(win, 0, 0, f"terminal too small (need {MIN_W}x{MIN_H})")
        win.refresh()
        return
    segs = em.segs
    safe_add(win, 0, 2, "E-METER", curses.A_BOLD)
    draw_knob(win, pal, rich, 2, "TONE ARM / SCROLL", em.ta_pos,
              f"{em.ta:5.2f} {UNIT}" if em.ta is not None else "  --",
              "A <  > D")
    draw_knob(win, pal, rich, 26, "SENSITIVITY / ZOOM", em.sens_pos,
              f"x{K / em.half:.1f}  ±{fmt(em.half)} {UNIT}",
              "K <  > L")
    draw_meter(win, pal, rich, Y_NOW, "NOW", em.now, segs, em.mode)
    draw_scale(win, Y_TICKS, em.half, segs, w)
    draw_meter(win, pal, rich, Y_AVG, "AVG", em.slow, segs, em.mode)
    safe_add(win, Y_STATUS, 1, em.status()[: w - 2], curses.A_DIM)
    safe_add(win, Y_HELP, 1,
             f"scale = {UNIT} from tone arm   Shift = coarse   s set   b needle/bar   q quit"[: w - 2],
             curses.A_DIM)
    win.refresh()


def setup_colors():
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    return bg, curses.COLORS >= 256


def ui(stdscr, args, q):
    curses.curs_set(0)
    stdscr.keypad(True)
    bg, rich = setup_colors()
    pal = Palette(bg)
    em = EMeter(q)
    next_tick = time.monotonic()

    while True:
        h, w = stdscr.getmaxyx()
        em.set_segments(dial_segments(w))

        stdscr.timeout(max(0, int((next_tick - time.monotonic()) * 1000)))
        ch = stdscr.getch()
        key = chr(ch) if 0 <= ch < 256 else ""
        if key in ("q", "Q"):
            return
        elif key in ("a", "A") or ch == curses.KEY_LEFT:
            em.turn_ta(-COARSE if key == "A" else -1)
        elif key in ("d", "D") or ch == curses.KEY_RIGHT:
            em.turn_ta(COARSE if key == "D" else 1)
        elif key in ("k", "K") or ch == curses.KEY_DOWN:
            em.turn_sens(-COARSE if key == "K" else -1)
        elif key in ("l", "L") or ch == curses.KEY_UP:
            em.turn_sens(COARSE if key == "L" else 1)
        elif key in ("s", "S"):
            em.set_needle()
        elif key in ("b", "B"):
            em.mode = "bar" if em.mode == "needle" else "needle"

        if time.monotonic() >= next_tick:
            em.tick()
            next_tick += TICK_MS / 1000.0
            if next_tick < time.monotonic():          # fell behind, don't burst
                next_tick = time.monotonic() + TICK_MS / 1000.0

        draw(stdscr, pal, rich, em)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="fake EDA signal, no hardware")
    ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=ADDR,
                    help="ADS1115 address (default 0x48)")
    args = ap.parse_args()

    locale.setlocale(locale.LC_ALL, "")               # needed for block characters
    q, stop = queue.Queue(), threading.Event()
    threading.Thread(target=reader_thread, daemon=True,
                     args=(q, stop, args.sim, args.bus, args.addr)).start()
    try:
        curses.wrapper(ui, args, q)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()