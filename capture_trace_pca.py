#!/usr/bin/env python3
"""
collect_profiling.py
Captures profiling traces from ChipWhisperer and saves:
 - traces_profile.csv
 - plaintexts_profile.csv
 - keys_profile.csv
"""

import os
import numpy as np
import chipwhisperer as cw

from typing import List

def read_hex_lines(filename: str, max_lines: int = None) -> List[bytes]:
    out = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            s = line.strip().replace(" ", "")
            if s:
                out.append(bytes.fromhex(s))
    return out

def program_target(scope, firmware_hex_path):
    try:
        prog = cw.programmers.STM32FProgrammer
        print("[*] Programming target:", firmware_hex_path)
        cw.program_target(scope, prog, firmware_hex_path)
    except Exception as e:
        print("[!] Programming failed (likely already programmed):", e)

def capture_traces_from_files(pt_file, key_file, n_traces, scope, target):
    pts = read_hex_lines(pt_file, max_lines=n_traces)
    keys = read_hex_lines(key_file, max_lines=n_traces)

    traces = []
    ptexts = []
    ktexts = []

    for i, (pt, key) in enumerate(zip(pts, keys)):
        trace = cw.capture_trace(scope, target, pt, key)
        if trace is None:
            print(f"[!] Missing trace {i}, skipping")
            continue

        wave = np.array(trace.wave)
        traces.append(wave)
        ptexts.append(list(pt))
        ktexts.append(list(key))

    return np.vstack(traces), np.array(ptexts), np.array(ktexts)

def main():
    profiling_pt_file = "pt_.txt"
    profiling_key_file = "key_.txt"
    firmware_hex = "simpleserial-aes-CW308_STM32F3.hex"
    n_profile_traces = 5000

    scope = cw.scope()
    scope.default_setup()
    target = cw.target(scope)

    if os.path.exists(firmware_hex):
        program_target(scope, firmware_hex)

    print("[*] Capturing profiling traces...")
    traces, pts, keys = capture_traces_from_files(
        profiling_pt_file,
        profiling_key_file,
        n_profile_traces,
        scope,
        target
    )

    print("[*] Saving CSV files...")
    np.savetxt("traces_profile.csv", traces, delimiter=",")
    np.savetxt("plaintexts_profile.csv", pts, fmt="%d", delimiter=",")
    np.savetxt("keys_profile.csv", keys, fmt="%d", delimiter=",")

    print("[+] Profiling data saved.")
    print("[+] Done.")

if __name__ == "__main__":
    main()
