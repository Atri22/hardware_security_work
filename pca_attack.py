#!/usr/bin/env python3
"""
perform_attack.py
Loads profile CSV files and performs the full ML-based attack.
Restores the original k-fold Pearson-r based POI selection.
"""

import numpy as np
from scipy.stats import pearsonr
import chipwhisperer as cw
import os

NUM_KEY_BYTES = 16

def load_profiling_data():
    traces = np.loadtxt("traces_profile.csv", delimiter=",")
    pts = np.loadtxt("plaintexts_profile.csv", delimiter=",", dtype=np.uint8)
    keys = np.loadtxt("keys_profile.csv", delimiter=",", dtype=np.uint8)
    return traces, pts, keys

def compute_variables(PLAINTEXTS, KEYS):
    VARIABLES = []
    for b in range(NUM_KEY_BYTES):
        VARIABLES.append([int(PLAINTEXTS[i][b]) ^ int(KEYS[i][b]) for i in range(len(PLAINTEXTS))])
    return VARIABLES

def classify(TRACES, VARIABLES):
    SETS = [[[] for _ in range(256)] for _ in range(NUM_KEY_BYTES)]
    for bnum in range(NUM_KEY_BYTES):
        for cla, trace in zip(VARIABLES[bnum], TRACES):
            SETS[bnum][cla].append(trace)
        for cla in range(256):
            if len(SETS[bnum][cla]) == 0:
                SETS[bnum][cla] = np.empty((0, TRACES.shape[1]))
            else:
                SETS[bnum][cla] = np.vstack(SETS[bnum][cla])
    return SETS

def estimate_means(SETS, nsamples):
    MEANS = np.zeros((NUM_KEY_BYTES, 256, nsamples))
    for bnum in range(NUM_KEY_BYTES):
        for cla in range(256):
            arr = SETS[bnum][cla]
            if arr.size > 0:
                MEANS[bnum][cla] = np.mean(arr, axis=0)
    return MEANS

#
# --- Restored POI-finding pipeline (k-fold, per-byte Pearson r) ---
#
def find_pois(TRACES, VARIABLES, k_fold, num_pois, poi_spacing):
    """
    Find POIs using k-fold cross-validated Pearson-r information measure.
    Returns POIS array shape (NUM_KEY_BYTES, num_pois) with sample indices.
    """
    num_samples = TRACES.shape[1]
    RS = estimate_r(TRACES, VARIABLES, k_fold)
    POIS = np.zeros((NUM_KEY_BYTES, num_pois), dtype=int)
    informative = RS.copy()   # shape (NUM_KEY_BYTES, num_samples)
    for bnum in range(NUM_KEY_BYTES):
        temp = informative[bnum].copy()
        for i in range(num_pois):
            poi = int(np.argmax(temp))
            POIS[bnum][i] = poi
            # zero-out neighborhood to enforce spacing
            pmin = max(0, poi - poi_spacing)
            pmax = min(num_samples, poi + poi_spacing)
            temp[pmin:pmax] = 0
    return POIS, RS

def estimate_r(TRACES, VARIABLES, k_fold):
    """
    Performs k-fold cross-validation:
      - For each fold: split profiling traces into profile/test
      - Compute MEANS_PROFILE (per-class means from profile set)
      - For each byte and each sample index compute Pearson r between
        TRACES_TEST[:,sample] and MEANS_TEST[b][:,sample]
      - Store RF[b,fold,sample] and average folds -> RS[b,sample]
    Returns RS (NUM_KEY_BYTES x num_samples)
    """
    num_samples = TRACES.shape[1]
    NUM = len(TRACES)
    RF = np.zeros((NUM_KEY_BYTES, k_fold, num_samples))
    for fold in range(k_fold):
        TRACES_PROFILE, TRACES_TEST, VARIABLES_PROFILE, VARIABLES_TEST = split(TRACES, VARIABLES, fold, k_fold)
        MEANS_PROFILE = classify_and_estimate_profile(TRACES_PROFILE, VARIABLES_PROFILE)
        MEANS_TEST = estimate_test(MEANS_PROFILE, TRACES_TEST, VARIABLES_TEST)  # shape (NUM_KEY_BYTES, ntest, num_samples)
        # For each byte and sample compute Pearson r between TRACES_TEST[:,i] and MEANS_TEST[b][:,i]
        for bnum in range(NUM_KEY_BYTES):
            for i in range(num_samples):
                # TRACES_TEST[:, i] has length ntest
                # MEANS_TEST[bnum][:, i] has length ntest
                try:
                    r, p = pearsonr(TRACES_TEST[:, i], MEANS_TEST[bnum][:, i])
                except Exception:
                    r = 0.0
                if np.isnan(r):
                    r = 0.0
                RF[bnum, fold, i] = r
    RS = np.average(RF, axis=1)  # average over folds -> shape (NUM_KEY_BYTES, num_samples)
    return RS

def split(TRACES, VARIABLES, fold, k_fold):
    """
    Deterministic k-fold split on profiling traces.
    Returns TRACES_PROFILE, TRACES_TEST, VARIABLES_PROFILE, VARIABLES_TEST
    """
    N = len(TRACES)
    Ntest = N // k_fold
    test_idx = list(range(fold * Ntest, min((fold + 1) * Ntest, N)))
    prof_idx = [i for i in range(N) if i not in test_idx]
    TRACES_PROFILE = TRACES[prof_idx]
    TRACES_TEST = TRACES[test_idx]
    VARIABLES_PROFILE = [[VARIABLES[b][i] for i in prof_idx] for b in range(NUM_KEY_BYTES)]
    VARIABLES_TEST = [[VARIABLES[b][i] for i in test_idx] for b in range(NUM_KEY_BYTES)]
    # convert VARIABLES_PROFILE to lists of same length as profiling/traces if needed
    return TRACES_PROFILE, TRACES_TEST, VARIABLES_PROFILE, VARIABLES_TEST

def classify_and_estimate_profile(TRACES_PROFILE, VARIABLES_PROFILE):
    """
    Build MEANS_PROFILE: shape (NUM_KEY_BYTES, 256, num_samples)
    from TRACES_PROFILE and VARIABLES_PROFILE.
    """
    num_samples = TRACES_PROFILE.shape[1]
    MEANS_PROFILE = np.zeros((NUM_KEY_BYTES, 256, num_samples))
    for bnum in range(NUM_KEY_BYTES):
        # build class buckets using profiling data
        sets = [[] for _ in range(256)]
        for cla, trace in zip(VARIABLES_PROFILE[bnum], TRACES_PROFILE):
            sets[cla].append(trace)
        for cla in range(256):
            arr = np.array(sets[cla])
            if arr.size == 0:
                # leave as zero
                continue
            MEANS_PROFILE[bnum, cla] = np.average(arr, axis=0)
    return MEANS_PROFILE

def estimate_test(MEANS_PROFILE, TRACES_TEST, VARIABLES_TEST):
    """
    For each test trace produce the corresponding class mean (from MEANS_PROFILE).
    Returns MEANS_TEST: shape (NUM_KEY_BYTES, ntest, num_samples)
    """
    num_samples = TRACES_TEST.shape[1]
    ntest = TRACES_TEST.shape[0]
    MEANS_TEST = np.zeros((NUM_KEY_BYTES, ntest, num_samples))
    for bnum in range(NUM_KEY_BYTES):
        for i in range(ntest):
            cl = VARIABLES_TEST[bnum][i]
            MEANS_TEST[bnum, i] = MEANS_PROFILE[bnum, cl]
    return MEANS_TEST

#
# --- End restored POI pipeline ---
#

def capture_attack_traces(pt_file, key_file, n, scope, target):
    pts = []
    traces = []
    fixed_key = open(key_file).read().strip().replace(" ", "")
    fixed_key = bytes.fromhex(fixed_key)

    with open(pt_file) as f:
        lines = [l.strip().replace(" ", "") for l in f.readlines()]

    for i in range(n):
        pt = bytes.fromhex(lines[i])
        trace = cw.capture_trace(scope, target, pt, fixed_key)
        if trace is None:
            raise RuntimeError(f"No trace captured for attack input {i}")
        traces.append(np.array(trace.wave))
        pts.append(list(pt))

    return np.vstack(traces), np.array(pts), fixed_key

def build_profile_from_means(MEANS, POIS):
    """
    Build PROFILE_MEANS[b, class, poi_index]
    """
    num_pois = POIS.shape[1]
    PROFILE_MEANS = np.zeros((NUM_KEY_BYTES, 256, num_pois))
    for bnum in range(NUM_KEY_BYTES):
        for cla in range(256):
            for i in range(num_pois):
                poi = int(POIS[bnum, i])
                PROFILE_MEANS[bnum, cla, i] = MEANS[bnum, cla, poi]
    return PROFILE_MEANS

def run_attack(PROFILE_MEANS, POIS, attack_traces, attack_pts):
    best = [0] * NUM_KEY_BYTES
    num_traces = attack_traces.shape[0]
    for b in range(NUM_KEY_BYTES):
        maxcpa = np.zeros(256)
        cols = POIS[b].astype(int)
        for kguess in range(256):
            clas = [(attack_pts[i][b] ^ kguess) for i in range(num_traces)]
            leaks = np.asarray([PROFILE_MEANS[b][cl] for cl in clas])  # shape (num_traces, num_pois)
            score = 1.0
            for i in range(leaks.shape[1]):
                try:
                    r, p = pearsonr(leaks[:, i], attack_traces[:, cols[i]])
                except Exception:
                    r = 0.0
                if np.isnan(r):
                    r = 0.0
                score *= r
            maxcpa[kguess] = score
        best[b] = int(np.argmax(maxcpa))
        print(f"[+] Subkey {b}: guess {best[b]} (score {maxcpa[best[b]]:.6g})")
    print("[+] Full key guess:", bytes(best).hex())
    return best

def main():
    print("[*] Loading profiling CSV data...")
    TRACES, PLAINTEXTS, KEYS = load_profiling_data()

    print("[*] Computing variables...")
    VARIABLES = compute_variables(PLAINTEXTS, KEYS)

    print("[*] Classifying (for full-profile MEANS)...")
    SETS = classify(TRACES, VARIABLES)
    MEANS = estimate_means(SETS, TRACES.shape[1])

    # POI selection settings (adjust as desired)
    k_fold = 10
    num_pois = 1
    poi_spacing = 5

    print("[*] Finding POIs (restored k-fold Pearson-r method)...")
    POIS, RS = find_pois(TRACES, VARIABLES, k_fold=k_fold, num_pois=num_pois, poi_spacing=poi_spacing)
    for b in range(NUM_KEY_BYTES):
        print(f" byte {b}: {POIS[b].tolist()}")

    PROFILE_MEANS = build_profile_from_means(MEANS, POIS)

    print("[*] Capturing attack traces...")
    scope = cw.scope()
    scope.default_setup()
    target = cw.target(scope)
    firmware_hex_path = "simpleserial-aes-CW308_STM32F3.hex"
    prog = cw.programmers.STM32FProgrammer
    print("[*] Programming target:", firmware_hex_path)
    cw.program_target(scope, prog, firmware_hex_path)
    attack_traces, attack_pts, fixed_key = capture_attack_traces("pt_attck.txt", "fixed_key.txt", 16, scope, target)

    print("[*] Running attack...")
    run_attack(PROFILE_MEANS, POIS, attack_traces, attack_pts)

if __name__ == "__main__":
    main()