#!/usr/bin/env python3
#

import os
import math
import numpy as np
from scipy.stats import pearsonr
import chipwhisperer as cw
import chipwhisperer.analyzer as cwa
import sys
from typing import List



# Globals
NUM_KEY_BYTES = 16
TRACES = None
PLAINTEXTS = None
KEYS = None
TRACES_REDUCED = None
VARIABLES = [None] * NUM_KEY_BYTES
SETS = None
MEANS = None
VARS = None
STDS = None
PROFILE_MEANS = None
PROFILE_STDS = None
PROFILE_COVS = None
POIS = None
MEANS_PROFILE = None
MEANS_TEST = None
LOG_PROBA = None

NUM_KEY_BYTES = 16
TRACES = None
PLAINTEXTS = None
KEYS = None
TRACES_REDUCED = None
VARIABLES = [None] * NUM_KEY_BYTES
SETS = None
MEANS = None
VARS = None
STDS = None
PROFILE_MEANS = None
PROFILE_STDS = None
PROFILE_COVS = None
POIS = None
MEANS_PROFILE = None
MEANS_TEST = None
LOG_PROBA = None

def normalize_trace(trace: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a single trace: (trace - mean) / std.
    If std is zero, return trace minus mean (so mean zero).
    """
    mu = np.mean(trace)
    sigma = np.std(trace)
    if sigma <= 0:
        return trace - mu
    return (trace - mu) / sigma

def read_hex_lines(filename: str, max_lines: int = None) -> List[bytes]:
    out = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            s = line.strip()
            if s == "":
                continue
            s = s.replace(" ", "")
            b = bytes.fromhex(s)
            if len(b) != 16:
                raise ValueError(f"Line {i+1} in {filename} is not 16 bytes (got {len(b)})")
            out.append(b)
    return out

def read_single_key(filename: str) -> bytes:
    with open(filename, "r") as f:
        s = f.readline().strip().replace(" ", "")
    b = bytes.fromhex(s)
    if len(b) != 16:
        raise ValueError("Fixed key file must contain a single 16-byte hex key")
    return b

def program_target(scope, firmware_hex_path):
    try:
        prog = cw.programmers.STM32FProgrammer
        print("[*] Programming target with:", firmware_hex_path)
        cw.program_target(scope, prog, firmware_hex_path)
    except Exception as e:
        print("[!] Programming target failed (continuing if already programmed):", e)
def capture_attack_traces(ptattack_file, fixed_key_file, n_traces, scope, target, project=None):
    pts = read_hex_lines(ptattack_file, max_lines=n_traces)
    fixed_key = read_single_key(fixed_key_file)
    traces = []
    ptexts = []
    for i, pt in enumerate(pts):
        trace = cw.capture_trace(scope, target, pt, fixed_key)
        if trace is None:
            print(f"[!] No trace captured for attack input {i}, skipping")
            continue
        wave = np.array(trace.wave)
        wave_norm = wave
        traces.append(wave_norm)
        ptexts.append(pt)
        if project is not None:
            project.traces.append(trace)
    if len(traces) == 0:
        raise RuntimeError("No attack traces captured")
    traces_np = np.vstack(traces)
    attack_plaintexts = np.array([list(b) for b in ptexts], dtype=np.uint8)
    print(f"[*] Captured {len(traces)} attack traces")
    return traces_np, attack_plaintexts, fixed_key
def load_traces_from_csv(traces_file, plaintext_file, key_file, n_traces=None):
    global TRACES, PLAINTEXTS, KEYS

    # ---- Load traces ----
    TRACES = np.loadtxt(traces_file, delimiter=",")
    if TRACES.ndim == 1:
        TRACES = TRACES.reshape(1, -1)  # handle 1-line files

    # ---- Load plaintexts ----
    PLAINTEXTS = np.loadtxt(plaintext_file, delimiter=",", dtype=np.uint8)
    if PLAINTEXTS.ndim == 1:
        PLAINTEXTS = PLAINTEXTS.reshape(1, -1)

    # ---- Load keys ----
    KEYS = np.loadtxt(key_file, delimiter=",", dtype=np.uint8)
    if KEYS.ndim == 1:
        KEYS = KEYS.reshape(1, -1)

    # ---- Limit to n_traces if specified ----
    if n_traces is not None:
        TRACES = TRACES[:n_traces]
        PLAINTEXTS = PLAINTEXTS[:n_traces]
        KEYS = KEYS[:n_traces]

    # ---- Safety checks ----
    if len(TRACES) != len(PLAINTEXTS) or len(TRACES) != len(KEYS):
        raise ValueError(
            f"Mismatch: traces={len(TRACES)}, pts={len(PLAINTEXTS)}, keys={len(KEYS)}"
        )

    print(f"[*] Loaded {len(TRACES)} traces from CSV, samples per trace = {TRACES.shape[1]}")
def compute_variables():
    global VARIABLES
    VARIABLE_FUNC = lambda p, k: p ^ k
    N = len(PLAINTEXTS)
    for bnum in range(NUM_KEY_BYTES):
        VARIABLES[bnum] = [VARIABLE_FUNC(int(PLAINTEXTS[i][bnum]), int(KEYS[i][bnum])) for i in range(N)]
def estimate():
    global MEANS, VARS, STDS
    num_samples = TRACES.shape[1]
    MEANS = np.zeros((NUM_KEY_BYTES, 256, num_samples))
   
    for bnum in range(NUM_KEY_BYTES):
        for cla in range(256):
            arr = SETS[bnum][cla]
            if arr.size == 0:
                continue
            MEANS[bnum][cla] = np.mean(arr, axis=0)
            
def classify():
    global SETS
    CLASSES = range(256)
    SETS = [[[] for _ in CLASSES] for _ in range(NUM_KEY_BYTES)]
    
    for bnum in range(NUM_KEY_BYTES):
        for cla, trace in zip(VARIABLES[bnum], TRACES):
            SETS[bnum][cla].append(trace)
        
        for cla in CLASSES:
            if len(SETS[bnum][cla]) == 0:
                SETS[bnum][cla] = np.empty((0, TRACES.shape[1]))
                print(f"⚠️  Key Byte {bnum}, Class {cla} is EMPTY")
            else:
                SETS[bnum][cla] = np.vstack(SETS[bnum][cla])

def find_pois(pois_algo, k_fold, num_pois, poi_spacing):
    global POIS, RS, PS, RZS, RF, PF
    num_samples = TRACES.shape[1]
    RS = np.zeros((NUM_KEY_BYTES, num_samples))
    RZS = np.zeros((NUM_KEY_BYTES, num_samples))
    POIS = np.zeros((NUM_KEY_BYTES, num_pois), dtype=int)
    estimate_r(k_fold)
    informative = RS
    for bnum in range(NUM_KEY_BYTES):
        temp = informative[bnum].copy()
        for i in range(num_pois):
            poi = int(np.argmax(temp))
            POIS[bnum][i] = poi
            pmin = max(0, poi - poi_spacing)
            pmax = min(len(temp), poi + poi_spacing)
            temp[pmin:pmax] = 0

def estimate_r(k_fold):
    global RS, RZS, PS, RF, PF, MEANS_PROFILE
    num_samples = TRACES.shape[1]
    PS = np.zeros((NUM_KEY_BYTES, num_samples))
    RF = np.zeros((NUM_KEY_BYTES, k_fold, num_samples))
    PF = np.zeros((NUM_KEY_BYTES, k_fold, num_samples))
    for fold in range(k_fold):
        split(fold, k_fold)
        classify_and_estimate_profile()
        estimate_test()
        estimate_rf_pf(fold)
    average_folds()

def split(fold, k_fold):
    global TRACES, VARIABLES
    N = len(TRACES)
    Ntest = N // k_fold
    test_idx = list(range(fold * Ntest, (fold + 1) * Ntest))
    prof_idx = [i for i in range(N) if i not in test_idx]
    TRACES_TEST = TRACES[test_idx]
    TRACES_PROFILE = TRACES[prof_idx]
    VARIABLES_TEST = [[VARIABLES[b][i] for i in test_idx] for b in range(NUM_KEY_BYTES)]
    VARIABLES_PROFILE = [[VARIABLES[b][i] for i in prof_idx] for b in range(NUM_KEY_BYTES)]
    globals().update({
        "TRACES_TEST": TRACES_TEST,
        "TRACES_PROFILE": TRACES_PROFILE,
        "VARIABLES_TEST": VARIABLES_TEST,
        "VARIABLES_PROFILE": VARIABLES_PROFILE
    })

def classify_and_estimate_profile():
    global MEANS_PROFILE
    CLASSES = range(256)
    MEANS_PROFILE = np.zeros((NUM_KEY_BYTES, len(CLASSES), TRACES.shape[1]))
    sets = [[[] for _ in CLASSES] for _ in range(NUM_KEY_BYTES)]
    for bnum in range(NUM_KEY_BYTES):
        for cla, trace in zip(VARIABLES_PROFILE[bnum], TRACES_PROFILE):
            sets[bnum][cla].append(trace)
        sets[bnum] = [np.array(sets[bnum][cla]) for cla in CLASSES]
    for bnum in range(NUM_KEY_BYTES):
        for cla in CLASSES:
            arr = sets[bnum][cla]
            
            if arr.size == 0:
                
                print(f"⚠️  Key Byte {bnum}, Class {cla} is EMPTY")
                # If no examples, leave mean as zero
                continue
            
            MEANS_PROFILE[bnum][cla] = np.average(arr, axis=0)

def estimate_test():
    global MEANS_TEST
    MEANS_TEST = np.zeros((NUM_KEY_BYTES, len(TRACES_TEST), TRACES.shape[1]))
    for bnum in range(NUM_KEY_BYTES):
        for i, trace in enumerate(TRACES_TEST):
            cl = VARIABLES_TEST[bnum][i]
            MEANS_TEST[bnum][i] = MEANS_PROFILE[bnum][cl]

def estimate_rf_pf(fold):
    global RF, PF
    num_samples = TRACES.shape[1]
    for bnum in range(NUM_KEY_BYTES):
        for i in range(num_samples):
            r, p = pearsonr(TRACES_TEST[:, i], MEANS_TEST[bnum][:, i])
            # if np.isnan(r):
            #     r = 0.0
            RF[bnum][fold][i] = r
           

def average_folds():
    global RS, PS, RF, PF
    RS = np.average(RF, axis=1)
    

def build_profile(num_pois):
    global PROFILE_MEANS, PROFILE_STDS, PROFILE_COVS
    PROFILE_MEANS = np.zeros((NUM_KEY_BYTES, 256, num_pois))
   
    for bnum in range(NUM_KEY_BYTES):
        for cla in range(256):
            for i in range(num_pois):
                poi = int(POIS[bnum][i])
                PROFILE_MEANS[bnum][cla][i] = MEANS[bnum][cla][poi]
                



def run_attack(attack_traces: np.ndarray, attack_plaintexts: np.ndarray):
    num_traces = attack_traces.shape[0]
    num_pois = POIS.shape[1]
    traces_reduced_attack = [None] * NUM_KEY_BYTES
    for bnum in range(NUM_KEY_BYTES):
        cols = POIS[bnum].astype(int)
        traces_reduced_attack[bnum] = attack_traces[:, cols]

    bestguess = [0] * NUM_KEY_BYTES
    LOG_PROBA = np.zeros((NUM_KEY_BYTES, 256))
    for bnum in range(NUM_KEY_BYTES):
        maxcpa = np.zeros(256)
        for kguess in range(256):
            clas = [(attack_plaintexts[i][bnum] ^ kguess) for i in range(num_traces)]
            leaks = np.asarray([PROFILE_MEANS[bnum][cl] for cl in clas])
            score = 1.0
            for i in range(leaks.shape[1]):
                r, p = pearsonr(leaks[:, i], traces_reduced_attack[bnum][:, i])
                if np.isnan(r):
                    r = 0.0
                score *= r
            maxcpa[kguess] = score
            LOG_PROBA[bnum][kguess] = score
        bestguess[bnum] = int(np.argmax(maxcpa))
        print(f"[+] Subkey {bnum} guess = {bestguess[bnum]} (score {maxcpa[bestguess[bnum]]:.6g})")
    print("[+] Full key guess:", bestguess)
    return bestguess

def main():
    # === settings: adapt as needed ===
    profiling_pt_file = "pt_.txt"
    profiling_key_file = "key_.txt"
    attack_pt_file = "pt_attck.txt"
    attack_fixed_key_file = "fixed_key.txt"
    firmware_hex = "simpleserial-aes-CW308_STM32F3.hex"
    output_project = "aes_project_norm.cwp"
    n_profile_traces = 5000
    n_attack_traces = 16
    pois_algo = "r"
    k_fold = 10
    num_pois = 1
    poi_spacing = 5

    scope = cw.scope()
    scope.default_setup()
    target = cw.target(scope)

    if os.path.exists(firmware_hex):
        program_target(scope, firmware_hex)
    else:
        print("[!] Firmware hex not found, skipping programming")

    project = cw.create_project(output_project, overwrite=True)
    print("[*] Capturing profiling traces (with normalization)...")
    load_traces_from_csv("traces_profile.csv", "plaintexts_profile.csv", "keys_profile.csv", n_traces=None)

    compute_variables()
    classify()
    estimate()
    print("[*] Finding POIs...")
    find_pois(pois_algo, k_fold, num_pois, poi_spacing)
    print("[*] POIs per byte:")
    for b in range(NUM_KEY_BYTES):
        print(f" byte {b}: {POIS[b].tolist()}")
    build_profile(num_pois)
    

    print("[*] Capturing attack traces (with normalization)...")
    attack_traces, attack_plaintexts, fixed_key = capture_attack_traces(attack_pt_file, attack_fixed_key_file, n_attack_traces, scope, target, project=project)

    print(f"[*] Saving project to {output_project}")
    project.save()

    print("[*] Running attack...")
    guessed = run_attack(attack_traces, attack_plaintexts)
    guessed_bytes = bytes(guessed)
    print("[+] Guessed key (bytes):", guessed_bytes.hex())

if __name__ == "__main__":
    main()
