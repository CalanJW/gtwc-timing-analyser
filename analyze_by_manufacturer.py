import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_seconds(time_str):
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
    if secs is None:
        return '—'
    m = int(secs // 60)
    rem = secs - m * 60
    if m > 0:
        return f"{m}:{rem:06.3f}"
    return f"{rem:.3f}"


def read_csv_semicolon(path):
    rows = []
    header = None
    with open(path, encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=';')
        for raw in reader:
            if header is None:
                header = [c.strip() for c in raw]
            elif raw and raw[0].strip() == header[0]:
                continue
            else:
                rows.append([c.strip() for c in raw])
    return header, rows


def manufacturer_from_car(car_name):
    """Extract manufacturer name from full car name string."""
    prefixes = [
        'Aston Martin',
        'Mercedes-AMG',
        'Lamborghini',
        'Chevrolet',
        'Ferrari',
        'Porsche',
        'McLaren',
        'Ford',
        'Audi',
        'BMW',
    ]
    for p in prefixes:
        if car_name.startswith(p):
            return p
    return car_name.split()[0]


# ── load sector data ────────────────────────────────────────────────────────────

sec_header, sec_rows = read_csv_semicolon('GTWCEU_GT3_R_SectorListCSV_1.0.CSV')
idx = {col: i for i, col in enumerate(sec_header)}

laps = []
bib_to_manufacturer = {}

for row in sec_rows:
    if len(row) < max(idx.values()) + 1:
        continue
    bib = row[idx['Bib']]
    lap = row[idx['Lap']]
    car = row[idx['Car']]
    try:
        lap_num = int(lap)
    except ValueError:
        continue
    s1 = parse_seconds(row[idx['Sector1Time']])
    s2 = parse_seconds(row[idx['Sector2Time']])
    s3 = parse_seconds(row[idx['Sector3Time']])
    laps.append({'bib': bib, 'lap': lap_num, 's1': s1, 's2': s2, 's3': s3})
    if bib not in bib_to_manufacturer:
        bib_to_manufacturer[bib] = manufacturer_from_car(car)

# ── load pit stop data ─────────────────────────────────────────────────────────

pit_header, pit_rows = read_csv_semicolon('GTWCEU_GT3_R_PitStopsCsv_1.0.CSV')
pidx = {col: i for i, col in enumerate(pit_header)}

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
    if not driver_out or driver_out.lower() == 'nan':
        driver_out = driver_in
    pits_by_bib[bib].append({'lap_in': lap_in, 'driver_in': driver_in, 'driver_out': driver_out})

for bib in pits_by_bib:
    pits_by_bib[bib].sort(key=lambda p: p['lap_in'])

# ── assign drivers to laps ─────────────────────────────────────────────────────

max_lap_by_bib = defaultdict(int)
for r in laps:
    if r['lap'] > max_lap_by_bib[r['bib']]:
        max_lap_by_bib[r['bib']] = r['lap']


def build_intervals(bib):
    stops = pits_by_bib.get(bib, [])
    if not stops:
        return []
    intervals = []
    current_driver = stops[0]['driver_in']
    current_start  = 1
    for pit in stops:
        intervals.append((current_start, pit['lap_in'], current_driver))
        current_driver = pit['driver_out']
        current_start  = pit['lap_in'] + 1
    intervals.append((current_start, max_lap_by_bib[bib] + 1, current_driver))
    return intervals


intervals_by_bib = {bib: build_intervals(bib) for bib in max_lap_by_bib}


def driver_for_lap(bib, lap_num):
    for start, end, drv in intervals_by_bib.get(bib, []):
        if start <= lap_num <= end:
            return drv
    return None


# ── global thresholds (same as overall scoreboard) ────────────────────────────

all_s1 = [r['s1'] for r in laps if r['s1'] is not None]
all_s2 = [r['s2'] for r in laps if r['s2'] is not None]
all_s3 = [r['s3'] for r in laps if r['s3'] is not None]

THRESHOLD = 1.40
max_s1 = min(all_s1) * THRESHOLD
max_s2 = min(all_s2) * THRESHOLD
max_s3 = min(all_s3) * THRESHOLD

# ── build per-manufacturer best sector times ───────────────────────────────────
# best_mfr[manufacturer][driver] = {s1, s2, s3, bib}

best_mfr = defaultdict(lambda: defaultdict(lambda: {'s1': None, 's2': None, 's3': None, 'bib': None}))

for lap_rec in laps:
    bib    = lap_rec['bib']
    mfr    = bib_to_manufacturer.get(bib)
    if not mfr:
        continue
    driver = driver_for_lap(bib, lap_rec['lap'])
    if driver is None:
        continue

    for key, threshold in [('s1', max_s1), ('s2', max_s2), ('s3', max_s3)]:
        val = lap_rec[key]
        if val is not None and val < threshold:
            rec = best_mfr[mfr][driver]
            if rec[key] is None or val < rec[key]:
                rec[key]  = val
                rec['bib'] = bib

# ── drawing function ───────────────────────────────────────────────────────────

def ranked_list_for(mfr_data, sector_key):
    entries = []
    for driver, data in mfr_data.items():
        if data[sector_key] is not None:
            entries.append({
                'driver':   driver,
                'bib':      data['bib'],
                'time':     data[sector_key],
                'time_str': fmt_time(data[sector_key]),
            })
    entries.sort(key=lambda e: e['time'])
    return entries


def draw_scoreboard(manufacturer, s1_ranked, s2_ranked, s3_ranked, out_path):
    N = max(len(s1_ranked), len(s2_ranked), len(s3_ranked), 1)

    ROW_H    = 0.36
    HEADER_H = 0.75
    TITLE_H  = 0.65
    FOOTER_H = 0.35
    COL_W    = 5.2
    MARGIN   = 0.30

    fig_w = COL_W * 3 + MARGIN * 2
    fig_h = TITLE_H + HEADER_H + N * ROW_H + FOOTER_H + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor('#0f0f1a')

    def y_frac(y): return y / fig_h
    def x_frac(x): return x / fig_w

    # Title
    fig.text(0.5, 1.0 - y_frac(TITLE_H * 0.35),
             f'{manufacturer.upper()} — BEST SECTOR TIMES',
             ha='center', va='center',
             fontsize=18, fontweight='bold',
             color='#ffffff', fontfamily='monospace')
    fig.text(0.5, 1.0 - y_frac(TITLE_H * 0.80),
             'Driver best sector — all stints',
             ha='center', va='center',
             fontsize=9, color='#888899', fontfamily='monospace')

    HDR_BG  = '#1e1e2e'
    ROW_ODD = '#1a1a2e'
    ROW_EVN = '#16213e'
    TXT     = '#ddddee'

    SECTOR_LABELS  = ['SECTOR 1', 'SECTOR 2', 'SECTOR 3']
    SECTOR_RANKED  = [s1_ranked, s2_ranked, s3_ranked]

    for col_idx, (label, ranked) in enumerate(zip(SECTOR_LABELS, SECTOR_RANKED)):
        col_x_left  = MARGIN + col_idx * COL_W
        col_x_right = col_x_left + COL_W - MARGIN * 0.5
        col_cx      = (col_x_left + col_x_right) / 2

        hdr_top    = fig_h - TITLE_H
        hdr_bottom = hdr_top - HEADER_H

        # Header band
        fig.add_artist(FancyBboxPatch(
            (x_frac(col_x_left), y_frac(hdr_bottom)),
            x_frac(col_x_right - col_x_left), y_frac(HEADER_H),
            boxstyle='square,pad=0', facecolor=HDR_BG, edgecolor='none',
            transform=fig.transFigure, clip_on=False
        ))
        fig.text(x_frac(col_cx), y_frac(hdr_bottom + HEADER_H * 0.62), label,
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 color='#ffffff', fontfamily='monospace')

        for sx, stxt in [
            (col_x_left + 0.32, 'POS'),
            (col_x_left + 0.82, '#'),
            (col_x_left + 1.45, 'DRIVER'),
            (col_x_right - 0.45, 'TIME'),
        ]:
            fig.text(x_frac(sx), y_frac(hdr_bottom + HEADER_H * 0.22), stxt,
                     ha='left' if stxt != 'TIME' else 'right', va='center',
                     fontsize=8, fontweight='bold', color='#888899',
                     fontfamily='monospace')

        # Data rows
        for rank_idx, entry in enumerate(ranked):
            row_y_top    = hdr_bottom - rank_idx * ROW_H
            row_y_bottom = row_y_top - ROW_H
            row_cy       = (row_y_top + row_y_bottom) / 2

            bg = ROW_ODD if rank_idx % 2 == 0 else ROW_EVN
            fig.add_artist(FancyBboxPatch(
                (x_frac(col_x_left), y_frac(row_y_bottom)),
                x_frac(col_x_right - col_x_left), y_frac(ROW_H),
                boxstyle='square,pad=0', facecolor=bg, edgecolor='none',
                transform=fig.transFigure, clip_on=False
            ))

            fs = 7.5
            fig.text(x_frac(col_x_left + 0.32), y_frac(row_cy),
                     f"P{rank_idx+1}", ha='left', va='center',
                     fontsize=fs, fontweight='bold', color=TXT, fontfamily='monospace')
            fig.text(x_frac(col_x_left + 0.82), y_frac(row_cy),
                     entry['bib'], ha='left', va='center',
                     fontsize=fs, color=TXT, fontfamily='monospace')
            name = entry['driver']
            if len(name) > 24:
                name = name[:23] + '…'
            fig.text(x_frac(col_x_left + 1.45), y_frac(row_cy),
                     name, ha='left', va='center',
                     fontsize=fs, color=TXT, fontfamily='monospace')
            fig.text(x_frac(col_x_right - 0.18), y_frac(row_cy),
                     entry['time_str'], ha='right', va='center',
                     fontsize=fs, fontweight='bold', color=TXT, fontfamily='monospace')

    # Separators
    for col_idx in range(1, 3):
        sep_x = x_frac(MARGIN + col_idx * COL_W - MARGIN * 0.25)
        fig.add_artist(plt.Line2D(
            [sep_x, sep_x], [y_frac(FOOTER_H), 1.0 - y_frac(TITLE_H * 0.1)],
            color='#333355', linewidth=0.5, transform=fig.transFigure
        ))

    fig.text(0.5, y_frac(FOOTER_H * 0.5),
             'Excludes formation/safety-car laps (>40 % above global field minimum per sector)  •  Sector times from official GTWC timing data',
             ha='center', va='center', fontsize=7, color='#555566',
             fontfamily='monospace')

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


# ── generate one PNG per manufacturer ─────────────────────────────────────────

os.makedirs('manufacturer_scoreboards', exist_ok=True)

for mfr in sorted(best_mfr.keys()):
    mfr_data   = best_mfr[mfr]
    s1_ranked  = ranked_list_for(mfr_data, 's1')
    s2_ranked  = ranked_list_for(mfr_data, 's2')
    s3_ranked  = ranked_list_for(mfr_data, 's3')
    safe_name  = mfr.replace(' ', '_').replace('-', '_')
    out_path   = f'manufacturer_scoreboards/{safe_name}.png'
    draw_scoreboard(mfr, s1_ranked, s2_ranked, s3_ranked, out_path)

print("Done.")
