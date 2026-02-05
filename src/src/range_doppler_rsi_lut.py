# -*- coding: utf-8 -*-
"""
Range–Doppler from IPG RSI + beam_steering LUT

What this script does
---------------------
1) Reads  RSI detection table (txt) and picks a suitable cycle (frame).
2) For each detection in that cycle, pulls 16 (Tx×Rx) steering phases from the LUT.
3) Synthesizes a real-valued IF data cube [N samples × Nc chirps × 4 Rx] by:
   - summing 4 Tx contributions per chirp (with per-chirp MIMO phase)
   - including range slope and Doppler terms
   - adding optional white Gaussian noise
4) Builds a Range–Doppler  with:
   - per-chirp DC removal (HPF)
   - strong range window (Blackman–Harris), Hann on Doppler
   - Doppler zero-pad for crisp bin spacing
   - Rx power-sum (display convenience)
   - optional CA-CFAR overlay to dim the background (visual)
5) NEW: saves a MATLAB .mat file with the IF cube and metadata for verification.

Inputs (edit below):
  - RSI text:  r"C:\Arjun Final Python\RSI_Output_2025-10-22_22-23-51.txt"
  - LUT text:  r"C:\Arjun Final Python\beam_steering_dummy.txt"
    (First run converts to a fast .bin file next to it)

Output:
  - PNG at     r"C:\Arjun Final Python\out\range_doppler_YYYYMMDD_HHMMSS.png"
  - MAT at     r"C:\Arjun Final Python\out\if_cube_c<cycle>_YYYYMMDD_HHMMSS.mat"

IMPORTANT: Algorithm/parameters match your current setup (MCR-2 dummy):
  * f0=76.5 GHz, B=600 MHz, fs=25 MHz, N=1024, Nc=320, Tchirp=40.96 µs
  * 16-pair steering LUT indexed by (az,el), 4×8 MIMO phase table
  * Blackman–Harris window (range), Hann window (Doppler), zero-padded Doppler FFT

Usage tip:
  - Set STAMPED_FILENAMES=True to always get a fresh output file.
  - Set SHOW_FIG=True to pop up the plot window for quick inspection.
"""

# "C:\\Arjun Final Python\\RSI_Output_2025-10-22-22-23-51.txt"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import savemat  # <-- added to export .mat

# ---------- Paths (edit these if needed) ----------
rsi_path = r"example_data/RSI_Output.txt"
lut_txt  = r"example_data/beam_steering_dummy.txt"
lut_bin  = os.path.splitext(lut_txt)[0] + ".bin"
out_dir  = "out"
os.makedirs(out_dir, exist_ok=True)
out_png  = os.path.join(out_dir, "range_doppler.png")

# ---------- Sensor & processing params (MCR-2 dummy) ----------
# Range resolution = c/(2B) = 0.25 m; we display first 448 bins ≈ 112 m.
c = 3.0e8
f0 = 76.5e9
lam = c / f0
B   = 600e6            # Hz
fs  = 25e6             # ADC rate
N   = 1024             # samples per chirp
Nc  = 320              # chirps per frame
Tchirp = 40.96e-6      # s  (matches N/fs exactly)
assert abs(Tchirp - (N/fs)) < 1e-9, "Tchirp must match N/fs to keep math consistent."

# Phase cycling for MIMO (radians) — same 4×8 table as createChirpSignal
# Each chirp uses one column; we sum the 4 Tx contributions within every chirp.
ChannelPhaseModulation = np.array([
    [0,0,0,0,0,0,0,0],
    [0,np.pi,0,np.pi,0,np.pi,0,np.pi],
    [0,np.pi/2,np.pi,3*np.pi/2,0,np.pi/2,np.pi,3*np.pi/2],
    [0,np.pi/4,np.pi/2,3*np.pi/4,np.pi,5*np.pi/4,3*np.pi/2,7*np.pi/4]
], dtype=np.float64)  # shape (4,8)

# ---------- Display / run-time controls ----------
PREFER_SINGLE_TARGET = True        # prefer a cycle with NumDetections==1 (clean vertical spike)
USE_CAFAR_OVERLAY    = False       # dim background using a simple CA-CFAR mask
DOPPLER_TARGET_NFFT  = 512         # Doppler zero-pad (512..1024 recommended)
SHOW_FIG             = True        # show plot window
STAMPED_FILENAMES    = True        # save a new file each run (prevents confusion with cached images)

# Reproducibility / noise controls
RNG_SEED             = 123456      # set to None for non-deterministic noise
NOISE_OFF            = False       # True → exact, noise-free baseline
DEFAULT_SNR_DB       = 5.0         # used when NOISE_OFF=False

# ---------- Helpers ----------
def blackman_harris(M: int) -> np.ndarray:
    """4-term Blackman–Harris (strong sidelobe suppression for range FFT)."""
    if M <= 1:
        return np.ones(M, dtype=float)
    n = np.arange(M)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    w = (a0
         - a1*np.cos(2*np.pi*n/(M-1))
         + a2*np.cos(4*np.pi*n/(M-1))
         - a3*np.cos(6*np.pi*n/(M-1)))
    return w.astype(float)

def hp_dc_per_chirp(adc: np.ndarray) -> np.ndarray:
    """
    Remove DC per chirp & per Rx channel (matches “turn on HPF” advice for display).
    adc: [Nsamples, Nchirps, Nrx]
    """
    return adc - adc.mean(axis=0, keepdims=True)

def simple_cafar_mask(power2d: np.ndarray, pfa=1e-3, guard=2, train=8) -> np.ndarray:
    """
    Tiny CA-CFAR (1D along Doppler) per range bin.
    For visualization only — this is not production CFAR.
    power2d: linear power [Nr, Nd] → boolean mask (True = keep/bright).
    """
    Nr, Nd = power2d.shape
    M = 2*train
    alpha = M*(pfa**(-1.0/M) - 1.0)

    mask = np.zeros_like(power2d, dtype=bool)
    k_train = np.ones(2*(train+guard)+1)
    k_guard = np.ones(2*guard+1)

    for r in range(Nr):
        row = power2d[r]
        # circular padding to avoid edge bias at Doppler edges
        pad = np.r_[row[-(train+guard):], row, row[:(train+guard)]]
        s_train = np.convolve(pad, k_train, mode='same')
        s_guard = np.convolve(pad, k_guard, mode='same')
        s_noise = s_train - s_guard
        s_noise = s_noise[(train+guard):(train+guard+Nd)]
        noise_est = s_noise / M
        thr = alpha * noise_est
        mask[r] = row > thr
    return mask

def load_rsi_table(path: str) -> pd.DataFrame:
    """
    Load RSI .txt exported by your toolchain.
    - Skips the fancy header (first line).
    - Assigns canonical column names used downstream.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"RSI file not found: {path}")
    cols = [
        "timestamp[s]",
        "numdetections",
        "x(range[m])",
        "y(azimuth[deg])",
        "z(elevation[deg])",
        "power[dbm]",
        "velocity[m/s]",
        "cycle",
    ]
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=cols,
        skiprows=1,
        dtype=float,
    )
    # Clean up types; ensure 'cycle' is integer-like
    for c_ in cols:
        df[c_] = pd.to_numeric(df[c_], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if not np.issubdtype(df["cycle"].dtype, np.integer):
        df["cycle"] = df["cycle"].round().astype(int)
    return df

def angle_to_index(angle_deg: np.ndarray) -> np.ndarray:
    """Clamp angle to [-90,90] and map to LUT grid index [0..1800] (0.1° step)."""
    a = np.clip(angle_deg, -90.0, 90.0)
    idx = np.rint((a + 90.0) / 180.0 * 1800.0).astype(int)
    return np.clip(idx, 0, 1800)

def get_lut_memmap(txt_path: str, bin_path: str) -> np.memmap:
    """
    Steering LUT: 16 phase values (Tx×Rx) for each (az, el).
    First run converts text→binary; subsequent runs memmap the .bin (fast).
    Shape: (1801, 1801, 16) with axes (azi, ele, txrx-pair).
    """
    shape = (1801, 1801, 16)
    if os.path.exists(bin_path):
        return np.memmap(bin_path, dtype=np.float32, mode="r", shape=shape)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"LUT text not found and no .bin present: {txt_path}")

    print("[LUT] Converting text -> binary (one-time). This can take several minutes…")
    arr = np.fromfile(txt_path, sep=" ", dtype=np.float64)
    if arr.size != np.prod(shape):
        raise ValueError(f"LUT size mismatch. Expected {np.prod(shape)}, got {arr.size}.")
    arr = arr.astype(np.float32, copy=False).reshape(shape, order="C")
    mm = np.memmap(bin_path, dtype=np.float32, mode="w+", shape=shape)
    mm[:] = arr[:]
    mm.flush()
    del mm
    print("[LUT] Saved binary:", bin_path)
    return np.memmap(bin_path, dtype=np.float32, mode="r", shape=shape)

def phases_for_objects(lut_mem: np.memmap,
                       az_deg: np.ndarray,
                       el_deg: np.ndarray) -> np.ndarray:
    """
    Pull per-detection steering phases and reshape to [nobj, 4, 4].
    Vandana's indexing rule: pair = tx*4 + rx.
    """
    azi_idx = angle_to_index(az_deg)
    ele_idx = angle_to_index(el_deg)
    phase16 = lut_mem[azi_idx, ele_idx, :]           # [nobj, 16]
    phase16 = np.asarray(phase16, dtype=np.float64)
    out = np.empty((phase16.shape[0], 4, 4), dtype=np.float64)
    for tx in range(4):
        for rx in range(4):
            out[:, tx, rx] = phase16[:, tx*4 + rx]
    return out

def synth_if_frame(power_mW: np.ndarray,
                   ranges_m: np.ndarray,
                   vel_mps: np.ndarray,
                   phases_txrx: np.ndarray,
                   SNRdB: float | None = DEFAULT_SNR_DB) -> np.ndarray:
    """
    Synthesize a real IF cube [N, Nc, 4] like createChirpSignal:
      - sum 4 Tx contributions per chirp per Rx
      - per-chirp MIMO phase (4×8 table)
      - fast-time range slope + inter-chirp Doppler term
      - optional white Gaussian noise scaled by sqrt(P)*10^(−SNR/20)
    """
    nobj = power_mW.size
    t_fast = np.arange(N) / fs
    k_slope = B / Tchirp

    rng = np.random.default_rng(RNG_SEED) if RNG_SEED is not None else np.random.default_rng()

    if SNRdB is None:  # noise-off baseline
        noise_sigma = np.zeros((nobj, 1), dtype=np.float64)
    else:
        noise_sigma = np.sqrt(power_mW) * (10**(-SNRdB/20.0))
        noise_sigma = noise_sigma[:, None]  # [nobj,1]

    IF = np.zeros((N, Nc, 4), dtype=np.float64)

    for cc in range(Nc):
        # Select the column of the 4×8 MIMO phase table
        phase_col = ChannelPhaseModulation[:, cc % ChannelPhaseModulation.shape[1]]  # [4]
        # Inter-chirp Doppler term (wrapped)
        doppler_mod = -np.mod(4.0*np.pi*f0/c * vel_mps * (cc*Tchirp), 2*np.pi)       # [nobj]

        for rx in range(4):
            acc_tx = np.zeros((nobj, N), dtype=np.float64)
            for tx in range(4):
                # Static steering + MIMO phase for this (obj, tx, rx)
                phi_txrx = phases_txrx[:, tx, rx] + phase_col[tx]                    # [nobj]
                # Fast-time slope: −(2π/c)*(2 v f0 + 2 r B/T)*t
                arg0 = -(2.0*np.pi/c)*(2.0*vel_mps*f0 + 2.0*ranges_m*k_slope)        # [nobj]
                # Full phase per object/sample
                arg = arg0[:, None]*t_fast[None, :] + phi_txrx[:, None] + doppler_mod[:, None]
                chirp = np.sqrt(power_mW/2.0)[:, None] * np.cos(arg)                 # [nobj,N]
                acc_tx += chirp
            # Add object-wise noise, then sum objects → IF[:, cc, rx]
            noise = noise_sigma * rng.standard_normal((nobj, N))
            IF[:, cc, rx] = (acc_tx + noise).sum(axis=0)

    return IF

# ---------- Windows & sizes for RD ----------
N_range_keep = 448                   # keep first 448 bins (~112 m)
win_range   = blackman_harris(N)     # strong range window
win_doppler = np.hanning(Nc)         # Doppler window

def rd_map_from_if(IF_cube: np.ndarray):
    """
    IF cube → RD heatmap (dB) and physical axes.
    Steps:
      1) DC removal per chirp (HPF)
      2) Range window + FFT (keep 0..447)
      3) Doppler window + zero-padded FFT (centered)
      4) Sum power across 4 Rx
      5) Optional CA-CFAR-style dimming for background
    Returns:
      RD_db_plot [Nr×Nd], rng_m [Nr], vel_mps [Nd]
    """
    # 1) HPF per chirp
    IF = hp_dc_per_chirp(IF_cube)

    # 2) Range window + FFT
    IF = win_range[:, None, None] * IF
    R = np.fft.fft(IF, n=N, axis=0)              # [N, Nc, 4]
    R = R[:N_range_keep, :, :]                   # keep near-field bins

    # 3) Doppler window + zero-padded FFT (then shift)
    Rw = R * win_doppler[None, :, None]
    Nd_fft = max(DOPPLER_TARGET_NFFT, 1 << ((Nc - 1).bit_length()))
    RD = np.fft.fftshift(np.fft.fft(Rw, n=Nd_fft, axis=1), axes=1)  # [448, Nd_fft, 4]

    # 4) Rx power sum
    RD_pow = (np.abs(RD) ** 2).sum(axis=2)       # [448, Nd_fft]

    # 5) Optional background dimming via simple CFAR mask
    if USE_CAFAR_OVERLAY:
        mask = simple_cafar_mask(RD_pow, pfa=1e-3, guard=2, train=8)
        RD_db = 10.0 * np.log10(np.maximum(RD_pow, 1e-12))
        bg_lvl = RD_db.min() + 0.2 * (RD_db.max() - RD_db.min())
        RD_db_plot = np.where(mask, RD_db, bg_lvl)
    else:
        RD_db_plot = 10.0 * np.log10(np.maximum(RD_pow, 1e-12))

    # Axes:
    rng_res = c / (2 * B)                        # 0.25 m for B=600 MHz
    rng_m   = np.arange(N_range_keep) * rng_res
    fD      = np.fft.fftshift(np.fft.fftfreq(Nd_fft, d=Tchirp))
    vel_mps = (lam / 2.0) * fD

    return RD_db_plot, rng_m, vel_mps

# ---------- Main ----------
def main():
    # 0) Sanity checks (clear error messages beat silent failures)
    if not os.path.exists(rsi_path):
        raise FileNotFoundError(f"RSI file not found: {rsi_path}")
    if not (os.path.exists(lut_txt) or os.path.exists(lut_bin)):
        raise FileNotFoundError(f"LUT not found (need .txt or .bin): {lut_txt}")

    # 1) Load RSI
    df = load_rsi_table(rsi_path)

    # 2) Choose cycle to plot
    if PREFER_SINGLE_TARGET:
        grp_det = df.groupby("cycle")["numdetections"].sum()
        single_cycles = grp_det[grp_det == 1].index.tolist()
        if len(single_cycles):
            cycle_to_plot = int(single_cycles[0])
        else:
            df["Power_lin"] = 10.0 ** (df["power[dbm]"] / 10.0)
            cycle_to_plot = int(df.groupby("cycle")["Power_lin"].sum().idxmax())
    else:
        df["Power_lin"] = 10.0 ** (df["power[dbm]"] / 10.0)
        cycle_to_plot = int(df.groupby("cycle")["Power_lin"].sum().idxmax())

    T = df[df["cycle"] == cycle_to_plot].copy()
    if T.empty:
        raise RuntimeError("Chosen cycle has no detections.")
    print(f"[INFO] Using Cycle={cycle_to_plot} with {len(T)} detection(s).")

    # 3) Load/prepare steering LUT
    lut = get_lut_memmap(lut_txt, lut_bin)  # (1801,1801,16) float32

    # 4) Build per-object phase tensor [nobj, 4, 4] from angles
    phases = phases_for_objects(
        lut_mem=lut,
        az_deg=T["y(azimuth[deg])"].to_numpy(dtype=np.float64),
        el_deg=T["z(elevation[deg])"].to_numpy(dtype=np.float64),
    )

    # 5) Synthesize IF cube
    snr_to_use = None if NOISE_OFF else DEFAULT_SNR_DB
    IF = synth_if_frame(
        power_mW=(10.0 ** (T["power[dbm]"] / 10.0)).to_numpy(dtype=np.float64),
        ranges_m=T["x(range[m])"].to_numpy(dtype=np.float64),
        vel_mps=T["velocity[m/s]"].to_numpy(dtype=np.float64),
        phases_txrx=phases,
        SNRdB=snr_to_use,
    )

    # 6) Range–Doppler map
    RD_db_plot, rng_m, vel_mps = rd_map_from_if(IF)

    # --- NEW: Save a MATLAB .mat for verification in MATLAB ---
    ts_mat = datetime.now().strftime("%Y%m%d_%H%M%S")
    mat_path = os.path.join(out_dir, f"if_cube_c{cycle_to_plot}_{ts_mat}.mat")
    mat_payload = {
        "signal_if_fullArray": IF.astype(np.float32),  # [N, Nc, 4]
        "fs": float(fs), "B": float(B), "f0": float(f0), "Tchirp": float(Tchirp),
        "N": int(N), "Nc": int(Nc), "NRx": 4,
        "range_m": rng_m.astype(np.float32),
        "doppler_vel_mps": vel_mps.astype(np.float32),
        "cycle": int(cycle_to_plot),
    }
    savemat(mat_path, mat_payload, do_compression=True)
    print("[OK] Saved IF cube (.mat) ->", mat_path)

    # 7) Save + show plot
    if STAMPED_FILENAMES:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(out_dir, f"range_doppler_c{cycle_to_plot}_{ts}.png")
    else:
        save_path = out_png

    plt.figure(figsize=(8, 5), dpi=130)
    extent = [vel_mps[0], vel_mps[-1], rng_m[-1], rng_m[0]]  # (vx_min, vx_max, r_max, r_min)
    plt.imshow(RD_db_plot, aspect="auto", extent=extent, origin="upper")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title(f"Range–Doppler (Cycle {cycle_to_plot})")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print("[OK] Saved Range–Doppler map ->", save_path)

    if SHOW_FIG:
        plt.show()
    plt.close()

if __name__ == "__main__":
    main()
