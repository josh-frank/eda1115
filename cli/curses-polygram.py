#!/usr/bin/env python3
"""
EDA polygraph, Stairmaster edition, curses version (works over plain SSH).

A rolling, mirrored bar-graph drawn with Unicode quarter/half/full blocks.
Every character cell is a 2x2 grid of sub-pixels (2 time samples wide,
2 levels tall), with a center line and bars growing up and down from it.

By default the graph fills your terminal. Data comes from an ADS1115 on
I2C via smbus2, or from a fake EDA signal with --sim.

Keys:
    Up / Down   more / less sensitive (switches to manual)
    a           toggle auto-squeeze
    r           re-center baseline on the current reading
    m           toggle signed (up/down) vs symmetric mirror
    q           quit
"""
import argparse
import curses
import locale
import math
import queue
import random
import threading
import time
from collections import deque

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


# --- display behaviour ----------------------------------------------------
TICK_MS = 100              # one new graph sample per tick
BASELINE_TAU_S = 20.0      # baseline follows slow drift over ~this long
MIN_FULLSCALE = 0.10 * K   # auto-squeeze never zooms in past this (µS, scaled by K)
LABEL_W = 7                # width of the scale labels on the left
GUTTER = LABEL_W + 1       # labels + axis column; graph starts after this

# (upper-left, upper-right, lower-left, lower-right) -> glyph
GLYPH = {
    (0, 0, 0, 0): " ",
    (1, 0, 0, 0): "▘",
    (0, 1, 0, 0): "▝",
    (0, 0, 1, 0): "▖",
    (0, 0, 0, 1): "▗",
    (1, 1, 0, 0): "▀",
    (0, 0, 1, 1): "▄",
    (1, 0, 1, 0): "▌",
    (0, 1, 0, 1): "▐",
    (1, 0, 0, 1): "▚",
    (0, 1, 1, 0): "▞",
    (1, 1, 1, 0): "▛",
    (1, 1, 0, 1): "▜",
    (1, 0, 1, 1): "▙",
    (0, 1, 1, 1): "▟",
    (1, 1, 1, 1): "█",
}


# --- rendering (pure functions) ---------------------------------------------
def render(devs, scale, half_rows, symmetric=False):
    """Signed deviations (uS, oldest first, len == 2*cols) -> text rows.

    Returns 2*half_rows + 1 strings: upper half, center line, lower half.
    `scale` is uS per sub-row; each half has 2*half_rows sub-rows.
    """
    cols = len(devs) // 2
    top = 2 * half_rows
    lv = [max(-top, min(top, round(d / scale))) for d in devs]

    def filled(s, dist, up):
        if symmetric:
            return int(dist <= abs(s))
        return int(dist <= s) if up else int(dist <= -s)

    def row(dist_top, dist_bot, up):
        out = []
        for c in range(cols):
            l, r = lv[2 * c], lv[2 * c + 1]
            out.append(GLYPH[(filled(l, dist_top, up), filled(r, dist_top, up),
                              filled(l, dist_bot, up), filled(r, dist_bot, up))])
        return "".join(out)

    rows = []
    for i in range(half_rows):                       # above center, top first
        d = 2 * (half_rows - i)
        rows.append(row(d, d - 1, True))
    rows.append("─" * cols)                          # center line
    for j in range(half_rows):                       # below center, nearest first
        rows.append(row(2 * j + 1, 2 * j + 2, False))
    return rows


def gradient(f):
    """0..1 -> (r, g, b): green / amber / red."""
    stops = [(0.0, (0, 220, 90)), (0.5, (255, 200, 0)), (1.0, (255, 50, 40))]
    f = max(0.0, min(1.0, f))
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f <= f1:
            t = (f - f0) / (f1 - f0)
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return stops[-1][1]


def row_fractions(half_rows):
    """Distance-from-center fraction (0..1) per row; None for the center line."""
    span = 2 * half_rows
    return ([(2 * (half_rows - i) - 0.5) / span for i in range(half_rows)]
            + [None]
            + [(2 * j + 1.5) / span for j in range(half_rows)])


def nice_step(min_step):
    """Smallest 1/2/5 x 10^n that is >= min_step."""
    exp = math.floor(math.log10(min_step))
    for mult in (1, 2, 5, 10):
        step = mult * 10 ** exp
        if step >= min_step * (1 - 1e-9):
            return step


def gutter(scale, half_rows, symmetric=False):
    """Left-hand scale: one (label, axis_char) per graph row.

    Ticks land on round uS values, at least 2 character rows apart, on the
    row where a bar of that value would end. Mirrors the render() rounding.
    """
    step = nice_step(4 * scale)
    dec = max(0, -math.floor(math.log10(step) + 1e-9))
    top = 2 * half_rows
    out = [("", "│")] * (2 * half_rows + 1)
    out[half_rows] = ("0", "┼")
    n = 1
    while True:
        m = round(n * step / scale)                  # sub-rows from center
        if m > top:
            break
        if m >= 1:
            v = n * step
            up_row = half_rows - math.ceil(m / 2)
            dn_row = half_rows + math.ceil(m / 2)
            out[up_row] = (f"{v:.{dec}f}" if symmetric else f"{v:+.{dec}f}", "┤")
            out[dn_row] = (f"{v:.{dec}f}" if symmetric else f"{-v:+.{dec}f}", "┤")
        n += 1
    return out


def rgb_to_256(rgb):
    r, g, b = (round(c / 255 * 5) for c in rgb)
    return 16 + 36 * r + 6 * g + b


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
            bumps.append((t, random.uniform(0.2, 0.9)))
            next_bump = t + random.uniform(6, 18)
        g = 2.5 + 0.35 * math.sin(t / 37) + 0.08 * math.sin(t / 5.3)
        for t_b, amp in bumps:
            dt = t - t_b
            g += amp * (1 - math.exp(-dt / 0.6)) * math.exp(-dt / 5.0)
        bumps = [(tb, a) for tb, a in bumps if t - tb < 40]
        q.put((g + random.gauss(0, 0.015)) * K)
        time.sleep(0.02)


def reader_thread(q, stop, sim, bus_no, addr):
    try:
        run_sim(q, stop) if sim else run_i2c(q, stop, bus_no, addr)
    except Exception as e:                       # surface errors in the UI
        q.put(f"{type(e).__name__}: {e}")


# --- state ------------------------------------------------------------------
class Polygraph:
    def __init__(self, q, cols, half_rows):
        self.q = q
        self.cols, self.H = 0, 0
        self.devs = deque()
        self.g = None
        self.baseline = None
        self.contact = True
        self.error = None
        self.auto = True
        self.symmetric = False
        self.fullscale = MIN_FULLSCALE          # current +/- range in uS
        self.alpha = (TICK_MS / 1000.0) / BASELINE_TAU_S
        self.resize(cols, half_rows)

    @property
    def scale(self):                                # uS per sub-row
        return self.fullscale / (2 * self.H)

    def resize(self, cols, half_rows):
        if (cols, half_rows) == (self.cols, self.H):
            return
        old = list(self.devs)[-2 * cols:]
        pad = [0.0] * (2 * cols - len(old))
        self.devs = deque(pad + old, maxlen=2 * cols)
        self.cols, self.H = cols, half_rows

    def zoom(self, factor):
        self.auto = False
        self.fullscale = max(MIN_FULLSCALE / 10, self.fullscale * factor)

    def recenter(self):
        if self.g is not None:
            self.baseline = self.g

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
            if self.baseline is None:
                self.baseline = self.g
            else:
                self.baseline += (self.g - self.baseline) * self.alpha
        elif bad:
            self.contact = False

        self.devs.append((self.g - self.baseline) if self.g is not None else 0.0)

        if self.auto:
            peak = max(abs(d) for d in self.devs)
            target = max(peak / 0.9, MIN_FULLSCALE)
            if target > self.fullscale:
                self.fullscale = target                              # squeeze in fast
            else:
                self.fullscale += (target - self.fullscale) * 0.05   # relax slowly

    def status(self):
        if self.error:
            return f"ERROR: {self.error}"
        if self.g is None:
            return "waiting for data..."
        msg = (f"{self.g:6.2f} {UNIT}  base {self.baseline:6.2f}  "
               f"full-scale +/-{fmt(self.fullscale)} {UNIT} "
               f"({'auto' if self.auto else 'manual'}, "
               f"{'symmetric' if self.symmetric else 'signed'})")
        if not self.contact:
            msg += "  [out of range, check contact]"
        return msg


# --- curses UI --------------------------------------------------------------
def setup_colors():
    """Start color support; return (background color, has 256 colors)."""
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    return bg, curses.COLORS >= 256


def build_attrs(half_rows, bg, rich):
    """Initialise one color pair per row for this layout; return attrs list."""
    attrs = []
    for idx, frac in enumerate(row_fractions(half_rows)):
        pair = idx + 1
        extra = 0
        if frac is None:                              # center line
            fg = 244 if rich else curses.COLOR_WHITE
            extra = 0 if rich else curses.A_DIM
        elif rich:
            fg = rgb_to_256(gradient(frac))
        else:
            fg = (curses.COLOR_GREEN if frac < 0.4 else
                  curses.COLOR_YELLOW if frac < 0.75 else curses.COLOR_RED)
        if pair < curses.COLOR_PAIRS:
            curses.init_pair(pair, fg, bg)
            attrs.append(curses.color_pair(pair) | extra)
        else:
            attrs.append(0)
    return attrs


def safe_add(win, y, x, s, attr=0):
    try:
        win.addstr(y, x, s, attr)
    except curses.error:
        pass                                          # ignore corner/overflow writes


def ui(stdscr, args, pg_q):
    curses.curs_set(0)
    stdscr.keypad(True)
    bg, rich = setup_colors()
    pg = None
    attrs, attrs_for = [], None
    next_tick = time.monotonic()

    while True:
        h, w = stdscr.getmaxyx()
        cols = max(4, min(args.cols or 10 ** 6, w - GUTTER - 2))
        half_rows = max(1, min(args.half_rows or 10 ** 6, (h - 4) // 2))
        if pg is None:
            pg = Polygraph(pg_q, cols, half_rows)
        pg.resize(cols, half_rows)
        if attrs_for != half_rows:
            attrs, attrs_for = build_attrs(half_rows, bg, rich), half_rows

        now = time.monotonic()
        stdscr.timeout(max(0, int((next_tick - now) * 1000)))
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return
        elif ch == curses.KEY_UP:
            pg.zoom(0.8)
        elif ch == curses.KEY_DOWN:
            pg.zoom(1.25)
        elif ch in (ord("a"), ord("A")):
            pg.auto = not pg.auto
        elif ch in (ord("r"), ord("R")):
            pg.recenter()
        elif ch in (ord("m"), ord("M")):
            pg.symmetric = not pg.symmetric

        if time.monotonic() >= next_tick:
            pg.tick()
            next_tick += TICK_MS / 1000.0
            if next_tick < time.monotonic():          # fell behind, don't burst
                next_tick = time.monotonic() + TICK_MS / 1000.0

        rows = render(list(pg.devs), pg.scale, pg.H, pg.symmetric)
        stdscr.erase()
        axis = gutter(pg.scale, pg.H, pg.symmetric)
        for i, line in enumerate(rows):
            label, tick = axis[i]
            safe_add(stdscr, i, 0, label[:LABEL_W].rjust(LABEL_W), attrs[i])
            safe_add(stdscr, i, LABEL_W, tick, curses.A_DIM)
            safe_add(stdscr, i, GUTTER, line, attrs[i])
        y = len(rows) + 1
        safe_add(stdscr, y, 1, pg.status()[: w - 2], curses.A_DIM)
        safe_add(stdscr, y + 1, 1,
                 f"scale = {UNIT} from baseline   up/down sensitivity  a auto  "
                 "r recenter  m signed/symmetric  q quit"[: w - 2],
                 curses.A_DIM)
        stdscr.refresh()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="fake EDA signal, no hardware")
    ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=ADDR,
                    help="ADS1115 address (default 0x48)")
    ap.add_argument("--cols", type=int, default=0,
                    help="max character columns (default: fill terminal width)")
    ap.add_argument("--half-rows", type=int, default=0,
                    help="max character rows above/below center (default: fill height)")
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