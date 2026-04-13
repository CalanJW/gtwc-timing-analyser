import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_seconds(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s or s.lower() == 'nan':
        return None
    try:
        if ':' in s:
            p = s.split(':')
            return int(p[0]) * 60 + float(p[1])
        return float(s)
    except (ValueError, IndexError):
        return None


def fmt_time(secs):
    if secs is None:
        return '—'
    m = int(secs // 60)
    r = secs - m * 60
    return f"{m}:{r:06.3f}" if m > 0 else f"{r:.3f}"


def read_csv_sc(path):
    rows, header = [], None
    with open(path, encoding='latin-1') as f:
        for raw in csv.reader(f, delimiter=';'):
            row = [c.strip() for c in raw]
            if header is None:
                header = row
            elif row and row[0] == header[0]:
                continue
            else:
                rows.append(row)
    return header, rows


def manufacturer_from_car(car):
    for p in ['Aston Martin', 'Mercedes-AMG', 'Lamborghini', 'Chevrolet',
              'Ferrari', 'Porsche', 'McLaren', 'Ford', 'Audi', 'BMW']:
        if car.startswith(p):
            return p
    return car.split()[0]


# ── load sector data ───────────────────────────────────────────────────────────

sec_hdr, sec_rows = read_csv_sc('GTWCEU_GT3_R_SectorListCSV_1.0.CSV')
sidx = {c: i for i, c in enumerate(sec_hdr)}

# (bib, lap) -> total lap time in seconds
lap_time = {}
bib_to_mfr = {}

for row in sec_rows:
    if len(row) <= max(sidx.values()):
        continue
    bib = row[sidx['Bib']]
    car = row[sidx['Car']]
    try:
        lap = int(row[sidx['Lap']])
    except ValueError:
        continue
    t = parse_seconds(row[sidx['Time']])
    if t is not None:
        lap_time[(bib, lap)] = t
    if bib not in bib_to_mfr:
        bib_to_mfr[bib] = manufacturer_from_car(car)

# ── load pit stop data ─────────────────────────────────────────────────────────

pit_hdr, pit_rows = read_csv_sc('GTWCEU_GT3_R_PitStopsCsv_1.0.CSV')
pidx = {c: i for i, c in enumerate(pit_hdr)}

# ── compute in-lap and out-lap times per driver ────────────────────────────────
#
# In-lap time  = sector-file total time for Lap In
#                (SF line → pit entry, naturally includes slow-down)
#
# Out-lap time = sector-file total time for (Lap In + 1) − nett pit time
#                The out-lap sector entry begins when the car enters the pit
#                lane, so the pit-stop duration is baked into the raw sector
#                time; subtracting nett time gives the on-track portion only.

# raw lists for threshold calculation
raw_in, raw_out = [], []

pit_events = []   # processed rows we'll iterate twice

for row in pit_rows:
    if len(row) <= max(pidx.values()):
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

    nett = parse_seconds(row[pidx['Nett Time']])

    t_in      = lap_time.get((bib, lap_in))
    t_out_raw = lap_time.get((bib, lap_in + 1))

    # Normal service stop (nett ≤ 100 s): the sector file starts the out-lap
    # clock from pit-ENTRY, so pit time is baked in → subtract nett.
    # Long / repair stop (nett > 100 s): the timing system resets at pit-EXIT,
    # so the sector file time is already the pure racing portion → use raw.
    if t_out_raw is not None and nett is not None:
        if nett <= 100:
            t_out = t_out_raw - nett
        else:
            t_out = t_out_raw
    else:
        t_out = None

    # sanity: out-lap must be a plausible racing time
    if t_out is not None and t_out < 85:
        t_out = None

    ev = dict(bib=bib, lap_in=lap_in,
              driver_in=driver_in, driver_out=driver_out,
              t_in=t_in, t_out=t_out)
    pit_events.append(ev)

    if t_in  is not None: raw_in.append(t_in)
    if t_out is not None: raw_out.append(t_out)

# threshold: 40 % above global fastest valid in/out lap
MIN_IN  = min(raw_in)
MIN_OUT = min(raw_out)
MAX_IN  = MIN_IN  * 1.40
MAX_OUT = MIN_OUT * 1.40

print(f"In-lap  fastest={fmt_time(MIN_IN)},  threshold<{fmt_time(MAX_IN)}")
print(f"Out-lap fastest={fmt_time(MIN_OUT)}, threshold<{fmt_time(MAX_OUT)}")

# best[driver] = {inlap, outlap, bib}
best = defaultdict(lambda: {'inlap': None, 'outlap': None, 'bib': None})

for ev in pit_events:
    bib       = ev['bib']
    driver_in = ev['driver_in']
    drv_out   = ev['driver_out']

    t_in  = ev['t_in']
    t_out = ev['t_out']

    if t_in is not None and t_in < MAX_IN and driver_in:
        rec = best[driver_in]
        if rec['inlap'] is None or t_in < rec['inlap']:
            rec['inlap'] = t_in
            rec['bib']   = bib

    if t_out is not None and t_out < MAX_OUT and drv_out:
        rec = best[drv_out]
        if rec['outlap'] is None or t_out < rec['outlap']:
            rec['outlap'] = t_out
            rec['bib']    = bib

# ── ranked lists ───────────────────────────────────────────────────────────────

def make_ranked(data, key):
    entries = []
    for driver, d in data.items():
        if d[key] is not None:
            entries.append({'driver': driver, 'bib': d['bib'],
                            'time': d[key], 'time_str': fmt_time(d[key])})
    entries.sort(key=lambda e: e['time'])
    return entries


# ── draw scoreboard ────────────────────────────────────────────────────────────

def draw_board(title, subtitle, inlap_ranked, outlap_ranked, out_path):
    N = max(len(inlap_ranked), len(outlap_ranked), 1)

    ROW_H    = 0.36
    HEADER_H = 0.75
    TITLE_H  = 0.65
    FOOTER_H = 0.35
    COL_W    = 5.2
    MARGIN   = 0.30

    fig_w = COL_W * 2 + MARGIN * 2
    fig_h = TITLE_H + HEADER_H + N * ROW_H + FOOTER_H + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor('#0f0f1a')

    def yf(y): return y / fig_h
    def xf(x): return x / fig_w

    # title
    fig.text(0.5, 1.0 - yf(TITLE_H * 0.35), title,
             ha='center', va='center', fontsize=18, fontweight='bold',
             color='#ffffff', fontfamily='monospace')
    fig.text(0.5, 1.0 - yf(TITLE_H * 0.80), subtitle,
             ha='center', va='center', fontsize=9,
             color='#888899', fontfamily='monospace')

    HDR_BG  = '#1e1e2e'
    ROW_ODD = '#1a1a2e'
    ROW_EVN = '#16213e'
    TXT     = '#ddddee'

    COL_LABELS  = ['IN LAP', 'OUT LAP']
    COL_RANKEDS = [inlap_ranked, outlap_ranked]

    for col_idx, (label, ranked) in enumerate(zip(COL_LABELS, COL_RANKEDS)):
        xl = MARGIN + col_idx * COL_W
        xr = xl + COL_W - MARGIN * 0.5
        cx = (xl + xr) / 2

        hdr_top = fig_h - TITLE_H
        hdr_bot = hdr_top - HEADER_H

        fig.add_artist(FancyBboxPatch(
            (xf(xl), yf(hdr_bot)), xf(xr - xl), yf(HEADER_H),
            boxstyle='square,pad=0', facecolor=HDR_BG, edgecolor='none',
            transform=fig.transFigure, clip_on=False
        ))
        fig.text(xf(cx), yf(hdr_bot + HEADER_H * 0.62), label,
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 color='#ffffff', fontfamily='monospace')

        for sx, stxt in [(xl+0.32,'POS'), (xl+0.82,'#'),
                         (xl+1.45,'DRIVER'), (xr-0.45,'TIME')]:
            fig.text(xf(sx), yf(hdr_bot + HEADER_H * 0.22), stxt,
                     ha='left' if stxt != 'TIME' else 'right', va='center',
                     fontsize=8, fontweight='bold', color='#888899',
                     fontfamily='monospace')

        for ri, entry in enumerate(ranked):
            yt = hdr_bot - ri * ROW_H
            yb = yt - ROW_H
            cy = (yt + yb) / 2

            bg = ROW_ODD if ri % 2 == 0 else ROW_EVN
            fig.add_artist(FancyBboxPatch(
                (xf(xl), yf(yb)), xf(xr - xl), yf(ROW_H),
                boxstyle='square,pad=0', facecolor=bg, edgecolor='none',
                transform=fig.transFigure, clip_on=False
            ))

            fs = 7.5
            fig.text(xf(xl+0.32), yf(cy), f"P{ri+1}",
                     ha='left', va='center', fontsize=fs, fontweight='bold',
                     color=TXT, fontfamily='monospace')
            fig.text(xf(xl+0.82), yf(cy), entry['bib'],
                     ha='left', va='center', fontsize=fs,
                     color=TXT, fontfamily='monospace')
            name = entry['driver']
            if len(name) > 24: name = name[:23] + '…'
            fig.text(xf(xl+1.45), yf(cy), name,
                     ha='left', va='center', fontsize=fs,
                     color=TXT, fontfamily='monospace')
            fig.text(xf(xr-0.18), yf(cy), entry['time_str'],
                     ha='right', va='center', fontsize=fs, fontweight='bold',
                     color=TXT, fontfamily='monospace')

    # separator
    sx = xf(MARGIN + COL_W - MARGIN * 0.25)
    fig.add_artist(plt.Line2D(
        [sx, sx], [yf(FOOTER_H), 1.0 - yf(TITLE_H * 0.1)],
        color='#333355', linewidth=0.5, transform=fig.transFigure
    ))

    fig.text(0.5, yf(FOOTER_H * 0.5),
             'In-lap = SF→pit-entry time  •  Out-lap = pit-exit→SF time  '
             '•  Outliers (>40 % above field fastest) excluded',
             ha='center', va='center', fontsize=7,
             color='#555566', fontfamily='monospace')

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


# ── overall scoreboard ─────────────────────────────────────────────────────────

os.makedirs('inout_scoreboards', exist_ok=True)

all_inlap  = make_ranked(best, 'inlap')
all_outlap = make_ranked(best, 'outlap')
print(f"Overall — in-lap drivers: {len(all_inlap)}, out-lap drivers: {len(all_outlap)}")

draw_board(
    'GTWC EUROPE — BEST IN & OUT LAPS',
    'Best in-lap and out-lap per driver — all manufacturers',
    all_inlap, all_outlap,
    'inout_scoreboards/overall.png'
)

# ── per-manufacturer scoreboards ───────────────────────────────────────────────

# Group best times by manufacturer (using the bib the best time was set on)
mfr_best = defaultdict(lambda: defaultdict(lambda: {'inlap': None, 'outlap': None, 'bib': None}))

for ev in pit_events:
    bib       = ev['bib']
    mfr       = bib_to_mfr.get(bib)
    if not mfr:
        continue

    driver_in = ev['driver_in']
    drv_out   = ev['driver_out']
    t_in      = ev['t_in']
    t_out     = ev['t_out']

    if t_in is not None and t_in < MAX_IN and driver_in:
        rec = mfr_best[mfr][driver_in]
        if rec['inlap'] is None or t_in < rec['inlap']:
            rec['inlap'] = t_in
            rec['bib']   = bib

    if t_out is not None and t_out < MAX_OUT and drv_out:
        rec = mfr_best[mfr][drv_out]
        if rec['outlap'] is None or t_out < rec['outlap']:
            rec['outlap'] = t_out
            rec['bib']    = bib

for mfr in sorted(mfr_best.keys()):
    data       = mfr_best[mfr]
    in_ranked  = make_ranked(data, 'inlap')
    out_ranked = make_ranked(data, 'outlap')
    safe       = mfr.replace(' ', '_').replace('-', '_')
    draw_board(
        f'{mfr.upper()} — BEST IN & OUT LAPS',
        'Best in-lap and out-lap per driver',
        in_ranked, out_ranked,
        f'inout_scoreboards/{safe}.png'
    )

print("Done.")
