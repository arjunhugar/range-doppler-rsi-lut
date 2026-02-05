# Range–Doppler from IPG RSI + Beam Steering LUT (Python)

## What this script does
1. Reads an RSI detection table and selects a suitable cycle (frame).
2. Pulls Tx×Rx steering phases from a beam-steering LUT.
3. Synthesizes a real-valued IF data cube `[N samples × Nc chirps × 4 Rx]`.
4. Builds a Range–Doppler heatmap using:
   - per-chirp DC removal
   - Blackman–Harris (range) and Hann (Doppler) windows
   - Doppler FFT zero-padding
   - Rx power sum for display
5. Saves:
   - PNG Range–Doppler plot
   - MATLAB `.mat` export for verification

## Repo structure
- `src/` : python script
- `example_data/` : placeholder (do not upload proprietary RSI/LUT files)
- `out/` : generated outputs (created at runtime)

## How to run
```bash
pip install -r requirements.txt
python src/range_doppler_rsi_lut.py
