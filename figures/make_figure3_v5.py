#!/usr/bin/env python3
"""Render aligned Figure 3 panels from byte-frozen plotted CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OLD = Path(__file__).resolve().parent
PREDECESSOR_SHA256 = 'c869e9693ecad4e425bddea462e495d8bcfbec508e5fc65173873f9fe5b315f3'
PLOTTED = {
    'events': ('position_mass_score_v3_events_plotted.csv', 'ec0ecea08c68ce912efa6e8ce5a0d09d110460bb22922e92574c7406f1e8d380'),
    'mass': ('position_mass_score_v3_mass_plotted.csv', '33f669e27751ccad1f754d97cb8654d54f63eb74a90c48d0542f593132974336'),
    'score': ('position_mass_score_v3_score_plotted.csv', 'd162da3e370f3e1a98d1bd58fba94d83b281a53e746eaf91ae148979fda2fb9c'),
}
ARMS = ('A00', 'A10', 'A01', 'A11', 'fp16_boundary', 'bf16_pair_boundary')
COL = {'A00': '#666666', 'A10': '#0072B2', 'A01': '#009E73', 'A11': '#CC79A7',
       'fp16_boundary': '#D55E00', 'bf16_pair_boundary': '#0072B2'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def series(rows, category, axis, value):
    result = {}
    for row in rows:
        result.setdefault(row[category], []).append((int(row[axis]), float(row[value])))
    for key in result:
        result[key].sort()
    return result


def make_plot(events, masses, scores, out: Path) -> dict:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12.5,
                         'axes.labelsize': 12.5, 'axes.titlesize': 13,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 12, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(7.3, 8.5))
    # Explicit bands keep each legend clear of its panel and tick labels.
    top = (fig.add_axes([.16, .76, .36, .14]),
           fig.add_axes([.61, .76, .36, .14]))
    mid = fig.add_axes([.16, .43, .81, .19])
    score_legend_axis = fig.add_axes([.16, .35, .81, .065])
    score_legend_axis.axis('off')
    bot = fig.add_axes([.16, .095, .81, .235])

    event_groups = [
        ('FP32 / S-BF16', 'A00', '#666666', '-', 'o'),
        ('Joint BF16', 'A11', '#CC79A7', '-', 's'),
        ('z-BF16', 'A01', '#009E73', '--', 'o'),
        ('FP16', 'fp16_boundary', '#D55E00', '-', '^'),
        ('Two-word BF16', 'bf16_pair_boundary', '#0072B2', '-', 'D'),
    ]
    event_map = {(r['arm'], int(r['layer']), int(r['step_start'])):
                 float(r['mean_image_absorbed_positive_rate']) for r in events}
    assert len(event_map) == len(events)
    mids = [63.5, 191.5, 383.5, 647]
    starts = [0, 128, 256, 512]
    for layer, ax in zip((0, 7), top):
        for label, arm, color, style, marker in event_groups:
            values = [event_map[(arm, layer, start)] for start in starts]
            ax.plot(mids, values, color=color, ls=style, marker=marker,
                    lw=1.8, markersize=5.5,
                    markerfacecolor='white' if arm == 'A01' else color,
                    markeredgewidth=1.5 if arm == 'A01' else .6, label=label)
        ax.set(title=f'Layer {layer}, head 0', ylim=(-.04, .94), xlim=(0, 783),
               xticks=[64, 192, 384, 647],
               xticklabels=['0–127', '128–255', '256–511', '512–782'])
        ax.grid(axis='y', color='#dddddd', lw=.7)
        ax.tick_params(axis='x', rotation=30)
    top[0].set_ylabel('Mean image fraction of\npositive increments absorbed')
    top[0].text(.02, .96, 'A', transform=top[0].transAxes,
                va='top', fontweight='bold', fontsize=13)
    handles, labels = top[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995),
               ncol=3, frameon=False, columnspacing=.8, handlelength=1.8)

    mass = series(masses, 'arm', 'completed_call_index',
                  'mean_image_ratio_of_selected_coordinate_sums_D_over_feed')
    assert set(mass) == set(ARMS) and all(len(v) == 783 for v in mass.values())
    endpoints = {arm: mass[arm][-1][1] for arm in ARMS}
    assert endpoints['A00'] == endpoints['A10'] == 0
    expected = {'A11': -10.23, 'A01': -10.27,
                'fp16_boundary': -.650, 'bf16_pair_boundary': -.00258}
    percent = {arm: 100 * value for arm, value in endpoints.items()}
    assert all(abs(percent[arm] - value) < (.005 if arm in ('A11', 'A01') else .0005)
               for arm, value in expected.items())
    mass_styles = [
        ('A11', 'Joint BF16  −10.23%', '#CC79A7', '-', 2.4, 3),
        ('A01', 'z-BF16  −10.27%', '#009E73', (0, (5, 2)), 2.2, 4),
        ('fp16_boundary', 'FP16  −0.650%', '#D55E00', '-', 2.1, 3),
        ('bf16_pair_boundary', 'Two-word BF16  −0.00258%', '#0072B2', (0, (1, 2)), 2.4, 5),
        ('A00', 'FP32 / S-BF16  0%', '#555555', (0, (4, 3)), 1.5, 2),
    ]
    for arm, label, color, style, width, zorder in mass_styles:
        x, y = zip(*mass[arm])
        mid.plot(x, y, label=label, color=color, ls=style, lw=width, zorder=zorder)
    common_ticks = [1, 128, 256, 384, 512, 640, 783]
    mid.set(xlim=(1, 783), ylim=(-.122, .014), xticks=common_ticks,
            ylabel='Mean image-level\nΣD / Σfeed')
    mid.text(.01, 1.035, 'B', transform=mid.transAxes,
             va='bottom', fontweight='bold', fontsize=13)
    mid.tick_params(labelbottom=False)
    mid.grid(axis='both', color='#dddddd', lw=.7)
    # Early mass curves stay above -0.04, leaving this lower-left region clear.
    handles, _ = mid.get_legend_handles_labels()
    endpoint_labels = ['Joint −10.23%', 'z-BF16 −10.27%', 'FP16 −0.650%',
                       'Pair −0.00258%', 'FP32/S-BF16 0%']
    mid.legend(handles, endpoint_labels, loc='lower left', bbox_to_anchor=(.01, .015),
               fontsize=11.5, frameon=True, facecolor='white', framealpha=.97,
               edgecolor='#bbbbbb', borderpad=.28, labelspacing=.18,
               handlelength=1.5, handletextpad=.5)

    score_names = {
        'A10_minus_A00': ('S-BF16 − FP32', COL['A10'], '-'),
        'A01_minus_A00': ('z-BF16 − FP32', COL['A01'], '-'),
        'A11_minus_A00': ('Joint BF16 − FP32', COL['A11'], '-'),
        'FP16_minus_A00': ('FP16 − FP32', COL['fp16_boundary'], '-'),
        'pair_minus_A00': ('Two-word − FP32', COL['bf16_pair_boundary'], '-'),
        'primary_A11_minus_A10': ('Joint BF16 − S-BF16', '#111111', '--'),
    }
    score = series(scores, 'contrast', 'predicted_pixel_index',
                   'mean_cumulative_bits_per_image')
    assert set(score) == set(score_names) and all(len(v) == 783 for v in score.values())
    for key, (label, color, style) in score_names.items():
        x, y = zip(*score[key])
        bot.plot(x, y, label=label, color=color, ls=style, lw=2.0)
    bot.axhline(0, color='#222222', lw=1.0)
    bot.set(xlim=(1, 783), xticks=common_ticks, xlabel='Predicted pixel index',
            ylabel='Mean cumulative score change\n(bits per image)')
    bot.text(.02, .96, 'C', transform=bot.transAxes,
             va='top', fontweight='bold', fontsize=13)
    bot.grid(axis='both', color='#dddddd', lw=.7)
    score_handles, score_labels = bot.get_legend_handles_labels()
    score_legend_axis.legend(score_handles, score_labels, loc='center',
                             ncol=3, frameon=False, fontsize=11.5,
                             columnspacing=.75, handlelength=1.8)

    fig.canvas.draw()
    mid_bounds = tuple(float(v) for v in mid.get_position().bounds)
    bot_bounds = tuple(float(v) for v in bot.get_position().bounds)
    mid_transform = [float(mid.transData.transform((x, 0))[0]) for x in (1, 128, 512, 783)]
    bot_transform = [float(bot.transData.transform((x, 0))[0]) for x in (1, 128, 512, 783)]
    assert abs(mid_bounds[0] - bot_bounds[0]) < 1e-12
    assert abs(mid_bounds[2] - bot_bounds[2]) < 1e-12
    assert mid.get_xlim() == bot.get_xlim() == (1.0, 783.0)
    assert list(mid.get_xticks()) == list(bot.get_xticks()) == common_ticks
    assert all(abs(a - b) < 1e-6 for a, b in zip(mid_transform, bot_transform))

    png = out / 'position_mass_score_v5.png'
    pdf = out / 'position_mass_score_v5.pdf'
    fig.savefig(png, dpi=300, facecolor='white')
    fig.savefig(pdf, facecolor='white', metadata={
        'CreationDate': datetime(2026, 9, 28, tzinfo=timezone.utc),
        'ModDate': datetime(2026, 9, 28, tzinfo=timezone.utc)})
    plt.close(fig)
    return {'output_png': png.name, 'output_pdf': pdf.name,
            'endpoint_percent_exact': percent,
            'width_inches': 7.3, 'height_inches': 8.5,
            'height_at_6_2in_width': 8.5 * 6.2 / 7.3,
            'smallest_legend_point_size_native': 11.5,
            'middle_axes_bounds': mid_bounds, 'bottom_axes_bounds': bot_bounds,
            'middle_x_transform': mid_transform, 'bottom_x_transform': bot_transform,
            'common_x_limits': list(mid.get_xlim()), 'common_x_ticks': common_ticks,
            'pixel_x_alignment_max_abs': max(abs(a-b) for a,b in zip(mid_transform,bot_transform))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error('refusing overwrite of existing output directory')
    out.mkdir(parents=True)
    start = time.monotonic()
    begin = datetime.now(timezone.utc).isoformat()
    source_hashes = {name: sha(OLD / name) for name, _ in PLOTTED.values()}
    csvs = {}
    for key, (name, expected_hash) in PLOTTED.items():
        source = OLD / name
        assert sha(source) == expected_hash
        copied = out / name.replace('_v3_', '_v5_')
        shutil.copyfile(source, copied)
        assert sha(copied) == expected_hash
        csvs[key] = {'source': str(source), 'output': copied.name,
                     'sha256': expected_hash, 'bytes': copied.stat().st_size}
    data = {key: read_csv(out / item['output']) for key, item in csvs.items()}
    details = make_plot(data['events'], data['mass'], data['score'], out)
    receipt = {'status': 'rendered_pending_visual_review',
               'started_utc': begin, 'ended_utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.monotonic() - start,
               'argv': [sys.executable, *sys.argv],
               'thread_environment': {k: os.environ.get(k) for k in
                                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
               'script_sha256': sha(Path(__file__)),
               'predecessor_script_sha256': PREDECESSOR_SHA256,
               'input_csv_sha256': source_hashes,
               'plotted_csv_byte_identity': csvs,
               'outputs_sha256': {n: sha(out / n) for n in
                                  (details['output_png'], details['output_pdf'])},
               'matplotlib_version': matplotlib.__version__,
               'details': details}
    (out / 'figure3_render_receipt_v1.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'output_dir': str(out),
                      'sha256': receipt['outputs_sha256']}))


if __name__ == '__main__':
    main()
