#!/usr/bin/env python3
"""Parse & visualize QNN runtime profile CSV data."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python analyze_profile.py <profile.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])  # the CSV written by Profiler::write_to_file()
os.makedirs("./logs", exist_ok=True)

# ── Layer latency breakdown ──────────────────────────────────────────────────
layer_data = df[df['name'].str.contains('layer_')]
if not layer_data.empty:
    layer_avg = layer_data.groupby('name')['duration_us'].mean().sort_values()

    plt.figure(figsize=(12, 6))
    layer_avg.plot(kind='barh')
    plt.title('Per-layer Latency Breakdown (μs)')
    plt.xlabel('Average duration (μs)')
    plt.tight_layout()
    plt.savefig('./logs/layer_breakdown.png', dpi=150)
    print("Saved layer_breakdown.png")
else:
    print("No layer-level data found in CSV.")

# ── Token throughput over time (shows thermal throttle) ─────────────────────
decode_data = df[df['name'] == 'decode_step_total']
if not decode_data.empty:
    decode_times = decode_data['duration_us']
    tps_series = 1e6 / decode_times  # convert μs/token → tokens/sec

    plt.figure(figsize=(14, 4))
    plt.plot(tps_series.values)
    plt.axhline(y=tps_series.mean(), color='r', linestyle='--',
                label=f'Mean: {tps_series.mean():.1f} tok/s')
    plt.axhline(y=tps_series.quantile(0.05), color='orange', linestyle='--',
                label=f'P5: {tps_series.quantile(0.05):.1f} tok/s')
    plt.title('Token Generation Speed Over Time (tok/s)')
    plt.xlabel('Token index')
    plt.ylabel('Tokens/sec')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./logs/throughput_timeline.png', dpi=150)
    print("Saved throughput_timeline.png")
else:
    print("No decode_step_total data found in CSV.")
    sys.exit(0)

# ── Print summary stats ──────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════")
print("INFERENCE PERFORMANCE SUMMARY")
print("═══════════════════════════════════════════")
print(f"Mean tok/s:    {tps_series.mean():.2f}")
print(f"Median tok/s:  {tps_series.median():.2f}")
print(f"P95 tok/s:     {tps_series.quantile(0.95):.2f}")
print(f"Min tok/s:     {tps_series.min():.2f}  ← throttle floor")
print(f"Max tok/s:     {tps_series.max():.2f}  ← peak burst")

htp_data = df[df['name'] == 'htp_graph_execute']['duration_us']
prep_data = df[df['name'] == 'decode_input_prep']['duration_us']
extract_data = df[df['name'] == 'decode_output_extract']['duration_us']
total_data = df[df['name'] == 'decode_step_total']['duration_us']

if not htp_data.empty and not total_data.empty:
    htp_time = htp_data.mean()
    prep_time = prep_data.mean() if not prep_data.empty else 0
    extract_time = extract_data.mean() if not extract_data.empty else 0
    total_time = total_data.mean()

    print(f"\nTime breakdown per token (mean):")
    print(f"  HTP execute:    {htp_time:.0f} μs  ({100*htp_time/total_time:.1f}%)")
    print(f"  Input prep:     {prep_time:.0f} μs  ({100*prep_time/total_time:.1f}%)")
    print(f"  Output extract: {extract_time:.0f} μs  ({100*extract_time/total_time:.1f}%)")
    print(f"  Total:          {total_time:.0f} μs")

print("═══════════════════════════════════════════")
