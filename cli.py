import argparse
import time
from smbus2 import SMBus   # `import smbus` works too: SMBus(1) is the same API

ADDR = 0x48                # ADS1115 default (ADDR pin to GND)
REG_CONV, REG_CONFIG = 0x00, 0x01

# Config: AIN0 vs GND, +/-4.096 V (same as Adafruit gain=1), continuous, 128 SPS, comparator off
CONFIG = 0x4000 | 0x0200 | 0x0000 | 0x0080 | 0x0003
LSB = 4.096 / 32768        # volts per count at this gain

R_REF, VCC = 100_000, 3.3

# ---- units ----
UNITS = {"u": ("uS", 1e6), "n": ("nS", 1e9)}
DEFAULT_UNIT = "u"         # "u" = microsiemens, "n" = nanosiemens; --unit overrides


def unit_arg(s):
    """Command-line value: u / us / µ / µs -> 'u', n / ns -> 'n'."""
    s = s.lower().rstrip("s")
    s = {"µ": "u", "μ": "u"}.get(s, s)         # micro sign and Greek mu
    if s not in UNITS:
        raise argparse.ArgumentTypeError("use u (uS) or n (nS)")
    return s


ap = argparse.ArgumentParser()
ap.add_argument("--unit", type=unit_arg, default=DEFAULT_UNIT, metavar="{u,n}",
                help="conductance unit: u = uS (default), n = nS")
ap.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
ap.add_argument("--addr", type=lambda s: int(s, 0), default=ADDR,
                help="ADS1115 address (default 0x48)")
args = ap.parse_args()

UNIT, SCALE = UNITS[args.unit]
K = SCALE / 1e6            # 1 for uS, 1000 for nS
W = 8 if K > 1 else 6      # reading column width

bus = SMBus(args.bus)
bus.write_i2c_block_data(args.addr, REG_CONFIG, [CONFIG >> 8, CONFIG & 0xFF])
time.sleep(0.02)


def read_voltage():
    hi, lo = bus.read_i2c_block_data(args.addr, REG_CONV, 2)   # ADS1115 is big-endian
    return int.from_bytes([hi, lo], "big", signed=True) * LSB


avg = None
try:
    while True:
        v = read_voltage()
        if 0.001 < v < VCC - 0.001:
            g = (VCC - v) / (v * R_REF) * SCALE                # uS or nS
            avg = g if avg is None else 0.8 * avg + 0.2 * g   # light smoothing
            bar = "#" * int(min(avg, 30 * K) * 2 / K)           # 0-30 uS (0-30000 nS) -> 0-60 chars
            line = f"{v:5.3f} V  {g:{W}.2f} {UNIT}  avg {avg:{W}.2f}  {bar}"
        else:
            line = f"{v:5.3f} V  out of range, check contact"
        print("\r" + line.ljust(100), end="", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print()
finally:
    bus.close()