import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_seconds(time_str):
    """Convert 'M:SS.mmm' or 'SS.mmm' to float seconds. Returns None on failure."""
    if not time_str or not isinstance(time_str, str):
        return None
    s = time_str.strip()
    if not s:
        return None
    try:
        if ':' in s:
            parts = s.split(':')
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, IndexError):
        return None


def fmt_time(secs):
    """Format float seconds back to 'M:SS.mmm' or 'SS.mmm'."""
    if secs is None:
        return '—'
    m = int(secs // 60)
    rem = secs - m * 60
    if m > 0:
        return f"{m}:{rem:06.3f}"
    return f"{rem:.3f}"


def read_csv_semicolon(path):
    """Read a semicolon-delimited CSV, skipping repeated header rows."""
    rows = []
    header = None
    with open(path, encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=';')
        for raw in reader:
            if header is None:
                header = [c.strip() for c in raw]
            elif raw and raw[0].strip() == header[0]:
                # repeated header row – skip
                continue
            else:
                rows.append([c.strip() for c in raw])
    return header, rows


# ── load sector data ────────────────────────────────────────────────────────────

sec_header, sec_rows = read_csv_semicolon('GTWCEU_GT3_R_SectorListCSV_1.0.CSV')
# Bib;Class;Driver1;Driver2;Driver3;Driver4;Car;Lap;Time;Sector1Time;SpeedTrap1;
#   Sector2Time;SpeedTrap2;Sector3Time;SpeedTrap3;TopSpeed
idx = {col: i for i, col in enumerate(sec_header)}

laps = []
for row in sec_rows:
    if len(row) < max(idx.values()) + 1:
        continue
    bib  = row[idx['Bib']]
    lap  = row[idx['Lap']]
    try:
        lap_num = int(lap)
    except ValueError:
        continue
    s1 = parse_seconds(row[idx['Sector1Time']])
    s2 = parse_seconds(row[idx['Sector2Time']])
    s3 = parse_seconds(row[idx['Sector3Time']])
    laps.append({'bib': bib, 'lap': lap_num, 's1': s1, 's2': s2, 's3': s3})

# ── load pit stop data ─────────────────────────────────────────────────────────

pit_header, pit_rows = read_csv_semicolon('GTWCEU_GT3_R_PitStopsCsv_1.0.CSV')
# Nr;Driver in;Day time in;Time in;Driver out;Day time out;Time out;Nett Time;Reason;Lap In
pidx = {col: i for i, col in enumerate(pit_header)}

# Group pit stops by car number, sorted by Lap In
pits_by_bib = defaultdict(list)
for row in pit_rows:
    if len(row) < max(pidx.values()) + 1:
        continue
    bib = row[pidx['Nr']]
    try:
        lap_in = int(float(row[pidx['Lap In']]))
    except (ValueError, IndexError):
        continue
    driver_in  = row[pidx['Driver in']].strip()
    driver_out = row[pidx['Driver out']].strip()
    if not driver_out or driver_out.lower() in ('', 'nan'):
        driver_out = driver_in
    pits_by_bib[bib].append({'lap_in': lap_in, 'driver_in': driver_in, 'driver_out': driver_out})

for bib in pits_by_bib:
    pits_by_bib[bib].sort(key=lambda p: p['lap_in'])

# ── assign driver to every lap ─────────────────────────────────────────────────

def build_intervals(bib, max_lap):
    """
    Return list of (start_lap, end_lap, driver_name) tuples.
    The starting driver is whoever appears as 'driver_in' of the first pit stop.
    """
    stops = pits_by_bib.get(bib, [])
    if not stops:
        return []

    intervals = []
    current_driver = stops[0]['driver_in']
    current_start  = 1

    for pit in stops:
        lap_in     = pit['lap_in']
        driver_out = pit['driver_out']
        intervals.append((current_start, lap_in, current_driver))
        current_driver = driver_out
        current_start  = lap_in + 1

    # final stint to end of race
    intervals.append((current_start, max_lap + 1, current_driver))
    return intervals


# Pre-compute max lap per bib
max_lap_by_bib = defaultdict(int)
for lap_rec in laps:
    if lap_rec['lap'] > max_lap_by_bib[lap_rec['bib']]:
        max_lap_by_bib[lap_rec['bib']] = lap_rec['lap']

intervals_by_bib = {bib: build_intervals(bib, max_lap_by_bib[bib])
                    for bib in max_lap_by_bib}

def driver_for_lap(bib, lap_num):
    for start, end, drv in intervals_by_bib.get(bib, []):
        if start <= lap_num <= end:
            return drv
    return None   # no pit data for this car


# ── compute best sector times per driver ───────────────────────────────────────

# First, get rough global min per sector to define a reasonable upper threshold
all_s1 = [r['s1'] for r in laps if r['s1'] is not None]
all_s2 = [r['s2'] for r in laps if r['s2'] is not None]
all_s3 = [r['s3'] for r in laps if r['s3'] is not None]

global_min_s1 = min(all_s1)
global_min_s2 = min(all_s2)
global_min_s3 = min(all_s3)

# Exclude sector times more than 40% above the global minimum
# (handles safety-car laps, formation laps, pit-lane exits)
THRESHOLD = 1.40
max_s1 = global_min_s1 * THRESHOLD
max_s2 = global_min_s2 * THRESHOLD
max_s3 = global_min_s3 * THRESHOLD

# best[driver][sector] = best_time
best = defaultdict(lambda: {'s1': None, 's2': None, 's3': None, 'bib': None})

for lap_rec in laps:
    bib     = lap_rec['bib']
    lap_num = lap_rec['lap']
    driver  = driver_for_lap(bib, lap_num)
    if driver is None:
        continue

    for key, threshold in [('s1', max_s1), ('s2', max_s2), ('s3', max_s3)]:
        val = lap_rec[key]
        if val is not None and val < threshold:
            if best[driver][key] is None or val < best[driver][key]:
                best[driver][key] = val
                best[driver]['bib'] = bib   # store car number too

# ── build ranked lists per sector ─────────────────────────────────────────────

def ranked_list(sector_key):
    entries = []
    for driver, data in best.items():
        if data[sector_key] is not None:
            entries.append({
                'driver': driver,
                'bib': data['bib'],
                'time': data[sector_key],
                'time_str': fmt_time(data[sector_key])
            })
    entries.sort(key=lambda e: e['time'])
    return entries

s1_ranked = ranked_list('s1')
s2_ranked = ranked_list('s2')
s3_ranked = ranked_list('s3')

print(f"Drivers with S1: {len(s1_ranked)}, S2: {len(s2_ranked)}, S3: {len(s3_ranked)}")
print(f"Global min — S1: {fmt_time(global_min_s1)}, S2: {fmt_time(global_min_s2)}, S3: {fmt_time(global_min_s3)}")
print("Top 5 S1:", [(e['driver'], e['time_str']) for e in s1_ranked[:5]])
print("Top 5 S2:", [(e['driver'], e['time_str']) for e in s2_ranked[:5]])
print("Top 5 S3:", [(e['driver'], e['time_str']) for e in s3_ranked[:5]])

# ── draw scoreboard ────────────────────────────────────────────────────────────

N = max(len(s1_ranked), len(s2_ranked), len(s3_ranked))
ROW_H  = 0.36          # inches per data row
HEADER_H = 0.75        # inches per sector-column header
TITLE_H  = 0.65        # inches for main title + subtitle
FOOTER_H = 0.35
COL_W  = 5.2           # inches per sector column
MARGIN = 0.30

fig_w = COL_W * 3 + MARGIN * 2
fig_h = TITLE_H + HEADER_H + N * ROW_H + FOOTER_H + 0.3

fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
fig.patch.set_facecolor('#0f0f1a')

# ── coordinate helpers ─────────────────────────────────────────────────────────
# We work in figure-fraction axes (0..1)
pw = fig_w     # page width in inches
ph = fig_h     # page height in inches

def y_frac(y_inch):
    return y_inch / ph

def x_frac(x_inch):
    return x_inch / pw

# ── title ─────────────────────────────────────────────────────────────────────
title_y = 1.0 - y_frac(TITLE_H * 0.35)
fig.text(0.5, title_y, 'GTWC EUROPE — BEST SECTOR TIMES',
         ha='center', va='center',
         fontsize=18, fontweight='bold',
         color='#ffffff', fontfamily='monospace')

subtitle_y = 1.0 - y_frac(TITLE_H * 0.80)
fig.text(0.5, subtitle_y, 'Driver best sector — all drivers, all stints',
         ha='center', va='center',
         fontsize=9, color='#888899', fontfamily='monospace')

# ── sector columns ─────────────────────────────────────────────────────────────

SECTOR_LABELS = ['SECTOR 1', 'SECTOR 2', 'SECTOR 3']
SECTOR_RANKED = [s1_ranked, s2_ranked, s3_ranked]

HDR_BG  = '#1e1e2e'   # uniform header background
ROW_ODD = '#1a1a2e'   # alternating row backgrounds
ROW_EVN = '#16213e'
TXT     = '#ddddee'

for col_idx, (label, ranked) in enumerate(zip(SECTOR_LABELS, SECTOR_RANKED)):
    # Column x positions
    col_x_left  = MARGIN + col_idx * COL_W
    col_x_right = col_x_left + COL_W - MARGIN * 0.5
    col_cx      = (col_x_left + col_x_right) / 2

    # Column header band
    hdr_top    = fig_h - TITLE_H
    hdr_bottom = hdr_top - HEADER_H
    hdr_rect = FancyBboxPatch(
        (x_frac(col_x_left), y_frac(hdr_bottom)),
        x_frac(col_x_right - col_x_left),
        y_frac(HEADER_H),
        boxstyle='square,pad=0',
        facecolor=HDR_BG, edgecolor='none',
        transform=fig.transFigure, clip_on=False
    )
    fig.add_artist(hdr_rect)

    fig.text(x_frac(col_cx), y_frac(hdr_bottom + HEADER_H * 0.62), label,
             ha='center', va='center',
             fontsize=14, fontweight='bold',
             color='#ffffff', fontfamily='monospace')

    # Sub-header labels
    col_sub_y = y_frac(hdr_bottom + HEADER_H * 0.22)
    sub_cols = [
        (col_x_left + 0.32, 'POS'),
        (col_x_left + 0.82, '#'),
        (col_x_left + 1.45, 'DRIVER'),
        (col_x_right - 0.45, 'TIME'),
    ]
    for sx, stxt in sub_cols:
        fig.text(x_frac(sx), col_sub_y, stxt,
                 ha='left' if stxt not in ('TIME',) else 'right',
                 va='center',
                 fontsize=8, fontweight='bold', color='#888899',
                 fontfamily='monospace')

    # Data rows
    data_top = hdr_bottom
    for rank_idx, entry in enumerate(ranked):
        row_y_top    = data_top - rank_idx * ROW_H
        row_y_bottom = row_y_top - ROW_H
        row_cy       = (row_y_top + row_y_bottom) / 2

        bg      = ROW_ODD if rank_idx % 2 == 0 else ROW_EVN
        txt_col = TXT

        rect = FancyBboxPatch(
            (x_frac(col_x_left), y_frac(row_y_bottom)),
            x_frac(col_x_right - col_x_left),
            y_frac(ROW_H),
            boxstyle='square,pad=0',
            facecolor=bg, edgecolor='none',
            transform=fig.transFigure, clip_on=False
        )
        fig.add_artist(rect)

        fsize = 7.5

        # Position number
        fig.text(x_frac(col_x_left + 0.32), y_frac(row_cy),
                 f"P{rank_idx + 1}",
                 ha='left', va='center',
                 fontsize=fsize, fontweight='bold', color=txt_col,
                 fontfamily='monospace')

        # Car number
        fig.text(x_frac(col_x_left + 0.82), y_frac(row_cy),
                 entry['bib'],
                 ha='left', va='center',
                 fontsize=fsize, color=txt_col,
                 fontfamily='monospace')

        # Driver name (truncate if too long)
        driver_name = entry['driver']
        if len(driver_name) > 24:
            driver_name = driver_name[:23] + '…'
        fig.text(x_frac(col_x_left + 1.45), y_frac(row_cy),
                 driver_name,
                 ha='left', va='center',
                 fontsize=fsize, color=txt_col,
                 fontfamily='monospace')

        # Time
        fig.text(x_frac(col_x_right - 0.18), y_frac(row_cy),
                 entry['time_str'],
                 ha='right', va='center',
                 fontsize=fsize, fontweight='bold', color=txt_col,
                 fontfamily='monospace')

# ── thin separator lines between columns ──────────────────────────────────────
for col_idx in range(1, 3):
    sep_x = x_frac(MARGIN + col_idx * COL_W - MARGIN * 0.25)
    line = plt.Line2D([sep_x, sep_x],
                      [y_frac(FOOTER_H), 1.0 - y_frac(TITLE_H * 0.1)],
                      color='#333355', linewidth=0.5,
                      transform=fig.transFigure)
    fig.add_artist(line)

# ── footer ─────────────────────────────────────────────────────────────────────
fig.text(0.5, y_frac(FOOTER_H * 0.5),
         'Excludes formation/safety-car laps (>40% above global minimum per sector)  •  Sector times from official GTWC timing data',
         ha='center', va='center',
         fontsize=7, color='#555566', fontfamily='monospace')

plt.savefig('sector_scoreboard.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved sector_scoreboard.png")
