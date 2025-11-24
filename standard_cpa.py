import chipwhisperer as cw
import numpy as np
import time

# ---------------- User params ----------------
PT_ATTACK_FILE = "pt_attck.txt"   # one 16-byte hex plaintext per line (32 hex chars)
KEY_FILE = "fixed_key.txt"          # one 16-byte hex key (32 hex chars)
NUM_TRACES = 50
SAVE_TRACES = True
TRACE_OUTFILE = "attack_traces.npz"
# ----------------------------------------------

# Local screenshot reference (the file you uploaded)
SCREENSHOT_URL = "file:///mnt/data/Screenshot 2025-11-22 184801.png"

# AES Sbox (kept here for CPA model/hamming)
SBOX = [
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
]

# ---------------- utilities ----------------
def read_hex_lines(filename, max_lines=None):
    pts = []
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if len(s) != 32:
                print(f"[!] Skipping invalid PT line: {s!r}")
                continue
            pts.append(bytes.fromhex(s))
            if max_lines is not None and len(pts) >= max_lines:
                break
    return pts

def read_single_key(filename):
    with open(filename, "r") as f:
        s = f.read().strip()
    if len(s) != 32:
        raise ValueError("Key file must contain exactly 32 hex chars (16 bytes)")
    return bytes.fromhex(s)

# ---------------- your simple capture function (unchanged style) ----------------
def capture_attack_traces(ptattack_file, fixed_key_file, n_traces, scope, target, project=None):
    pts = read_hex_lines(ptattack_file, max_lines=n_traces)
    fixed_key = read_single_key(fixed_key_file)

    traces = []
    ptexts = []

    print("[*] Starting simple capture_attack_traces()")

    for i, pt in enumerate(pts):
        print(f"[*] Capture {i+1}/{len(pts)} ...")

        # use cw.capture_trace exactly as you specified
        trace = cw.capture_trace(scope, target, pt, fixed_key)

        if trace is None:
            print(f"[!] No trace captured for attack input {i}, skipping")
            continue

        # support both Trace-like objects and raw lists returned by some CW versions
        if hasattr(trace, "wave"):
            wave = np.array(trace.wave, dtype=float)
        else:
            wave = np.array(trace, dtype=float)

        traces.append(wave)
        ptexts.append(pt)

        if project is not None:
            project.traces.append(trace)

    if len(traces) == 0:
        raise RuntimeError("No attack traces captured")

    traces_np = np.vstack(traces)
    attack_plaintexts = np.array([list(b) for b in ptexts], dtype=np.uint8)
    print(f"[*] Captured {len(traces)} attack traces")
    return traces_np, attack_plaintexts, fixed_key

# ---------------- CPA routine ----------------
def cpa_recover_key(traces_np, attack_plaintexts):
    """
    traces_np: (N, samples) numpy array
    attack_plaintexts: (N, 16) uint8 numpy array
    returns recovered_key list of 16 bytes
    """
    N, num_samples = traces_np.shape
    recovered = [0] * 16

    for byte in range(16):
        print(f"[CPA] Attacking byte {byte} ...")
        pt_bytes = attack_plaintexts[:, byte].astype(int)

        # compute correlation matrix for each key guess x sample
        correlations = np.zeros((256, num_samples), dtype=float)

        for k in range(256):
            # hypothetical HW vector (N,)
            hypo = np.array([bin(SBOX[pt_bytes[i] ^ k]).count("1") for i in range(N)], dtype=float)
           

            # for each sample, compute correlation between hypo and trace column
            for s in range(num_samples):
                trace_col = traces_np[:, s].astype(float)
                
                # np.corrcoef returns 2x2 matrix; [0,1] is r
                r = np.corrcoef(hypo, trace_col)[0, 1]
                if np.isnan(r):
                    r = 0.0
                correlations[k, s] = r

        # find key guess with maximum absolute correlation across all samples
        idx = np.nanargmax(np.abs(correlations))
        best_key = idx // num_samples
        recovered[byte] = int(best_key)
        print(f"[CPA] byte {byte} -> 0x{best_key:02x}")

    return recovered

# ---------------- main flow ----------------
def main():
    scope = cw.scope()
    scope.default_setup()
    target = cw.target(scope)
    firmware_hex_path='simpleserial-aes-CW308_STM32F3.hex'
    try:
        prog = cw.programmers.STM32FProgrammer
        print("[*] Programming target with:", firmware_hex_path)
        cw.program_target(scope, prog, firmware_hex_path)
    except Exception as e:
        print("[!] Programming target failed (continuing if already programmed):", e)

    # capture using your simple function (uses cw.capture_trace)
    traces_np, attack_plaintexts, fixed_key = capture_attack_traces(
        PT_ATTACK_FILE, KEY_FILE, NUM_TRACES, scope, target
    )

    # optional: save traces
    if SAVE_TRACES:
        np.savez(TRACE_OUTFILE, traces=traces_np, plaintexts=attack_plaintexts, key=np.frombuffer(fixed_key, dtype=np.uint8))
        print(f"Saved traces to {TRACE_OUTFILE}")

    # run CPA
    recovered = cpa_recover_key(traces_np, attack_plaintexts)
    print("\nRecovered AES-128 key (hex):", "".join(f"{b:02x}" for b in recovered))
    print("Reference screenshot (local):", SCREENSHOT_URL)

if __name__ == "__main__":
    main()
