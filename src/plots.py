"""Figures (PDF) from <out_dir>/summary.json and raw result files. Fixed categorical order per method
(validated reference palette: blue, orange, aqua, yellow, magenta). Usage: python src/plots.py --cfg CFG"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomark.common import load_cfg, p

COL = {'gm_kgw': '#2a78d6', 'gm_gumbel': '#eb6834', 'exact_kgw': '#1baf7a', 'posthoc': '#eda100', 'wtrace': '#e87ba4'}
LAB = {'gm_kgw': 'GeoMark-KGW-2R', 'gm_gumbel': 'GeoMark-Gumbel-H', 'exact_kgw': 'Exact-hash KGW', 'posthoc': 'Post-hoc (same hash)', 'wtrace': 'W-Trace'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e5e5e3',
                     'grid.linewidth': 0.5, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'lines.linewidth': 1.5,
                     'legend.frameon': False, 'pdf.fonttype': 42})


def save(fig, d, name):
    os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(d, name + '.pdf'))
    plt.close(fig)
    print('wrote', os.path.join(d, name + '.pdf'))


def fig_qq(cfg, S, d):
    fs = sorted(glob.glob(os.path.join(cfg['out_dir'], 'calibration', '*_real.npz')))
    if not fs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, f in enumerate(fs[:4]):
        z = np.load(f)
        pv = np.sort(10 ** z['lp_rand'])
        e = (np.arange(1, len(pv) + 1) - 0.5) / len(pv)
        ax.loglog(e, pv, lw=1.2, color=list(COL.values())[i % 5], label=os.path.basename(f)[:-9].replace('_', ' '))
    ax.loglog([1e-5, 1], [1e-5, 1], color='#52514e', lw=0.8, ls='--')
    ax.set_xlabel('expected p (uniform)'); ax.set_ylabel('observed null p (real traces, random keys)')
    ax.legend(fontsize=6)
    save(fig, d, 'qq_calibration')


def fig_tpr_vs_pairs(S, d):
    tv = S.get('tpr_vs_pairs', {})
    gens = sorted({k.split('|')[0] for k in tv})
    for gen in gens:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
        for ax, atk in zip(axes, ['none', 'mm']):
            for wm in ['gm_kgw', 'gm_gumbel', 'exact_kgw', 'posthoc']:
                r = tv.get(f'{gen}|{wm}|{atk}')
                if not r:
                    continue
                xs = [float(b.strip('[)').split(',')[0]) for b in r]
                ax.plot(xs, [v['tpr_1e3'] for v in r.values()], marker='o', ms=3, color=COL[wm], label=LAB[wm])
            ax.set_title(f'attack: {atk}'); ax.set_xlabel('distinct scored (seed, cell) pairs')
        axes[0].set_ylabel('TPR at FPR 1e-3'); axes[0].legend(fontsize=6)
        save(fig, d, f'tpr_vs_pairs_{gen}')


def fig_attacks(S, d):
    det = S.get('detection', {})
    gens = sorted({k.split('|')[0] for k in det})
    for gen in gens:
        atks = []
        for k in det:
            g, wm, atk = k.split('|')[:3]
            if g == gen and wm == 'gm_kgw' and atk not in atks:
                atks.append(atk)
        if not atks:
            continue
        wms = [w for w in ['gm_kgw', 'gm_gumbel', 'exact_kgw', 'posthoc'] if any(k.startswith(f'{gen}|{w}|') for k in det)]
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        w = 0.8 / len(wms)
        for i, wm in enumerate(wms):
            v = [det.get(f'{gen}|{wm}|{a}', {}).get('tpr_1e3', [np.nan])[0] for a in atks]
            ax.bar(np.arange(len(atks)) + i * w, v, w * 0.92, color=COL[wm], label=LAB[wm])
        ax.set_xticks(np.arange(len(atks)) + 0.4 - w / 2); ax.set_xticklabels(atks, rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('per-trace TPR at FPR 1e-3'); ax.set_ylim(0, 1); ax.legend(fontsize=6, ncol=4)
        save(fig, d, f'attacks_{gen}')


def fig_survival(S, d):
    det = S.get('detection', {})
    rows = [(k, v) for k, v in det.items() if 'lemma3_pred_surv_f' in v and 'surv_f' in v]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for k, v in rows:
        wm = k.split('|')[1]
        ax.scatter(v['lemma3_pred_surv_f'], v['surv_f'][0], s=12, color=COL.get(wm, '#4a3aa7'))
    ax.plot([0, 1], [0, 1], color='#52514e', lw=0.8, ls='--')
    ax.set_xlabel('Lemma 3 predicted surviving fraction'); ax.set_ylabel('measured surviving fine pairs')
    save(fig, d, 'survival_lemma3')


def fig_pooled(S, d):
    po = S.get('pooled', {})
    gens = sorted({k.split('|')[0] for k in po})
    for gen in gens:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for wm in ['gm_kgw', 'gm_gumbel', 'exact_kgw']:
            for atk, ls in [('none', '-'), ('mm', '--')]:
                pts = sorted((int(k.split('|m')[-1]), v['median_lp']) for k, v in po.items()
                             if k.startswith(f'{gen}|{wm}|{atk}|-|'))
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], ls=ls, marker='o', ms=3, color=COL[wm], label=f'{LAB[wm]} ({atk})')
        ax.set_xscale('log'); ax.set_xlabel('traces in release'); ax.set_ylabel('median log10 p (pooled)')
        ax.legend(fontsize=5)
        save(fig, d, f'pooled_{gen}')


def fig_steal(S, d):
    st = S.get('steal', {})
    corp = sorted({k.split('|')[0] for k in st})
    if not corp:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for i, c in enumerate(corp):
        pts = sorted((v['N'], v) for k, v in st.items() if k.startswith(c + '|N'))
        if not pts:
            continue
        col = list(COL.values())[i % 5]
        kk = [k for k in pts[0][1]['spoof']][0]
        axes[0].plot([n for n, _ in pts], [v['spoof'][kk]['frac_lp_le_m3'] for _, v in pts], marker='o', ms=3, color=col, label=c)
        rates = [k for k in pts[-1][1] if k.startswith('scrub_r')]
        ref = st.get(c + '|ref', {})
        axes[1].plot([pts[-1][1][r]['frac_changed'] for r in rates], [pts[-1][1][r]['median_lp'] for r in rates], marker='o', ms=3, color=col, label=f'{c} scrub')
        rr = [k for k in ref if k.startswith('scrub_random_r')]
        if rr:
            axes[1].plot([ref[r]['frac_changed'] for r in rr], [ref[r]['median_lp'] for r in rr], marker='x', ms=4, ls=':', color=col, label=f'{c} random')
    axes[0].set_xscale('log'); axes[0].set_xlabel('watermarked traces observed (N)'); axes[0].set_ylabel('spoof success (p<1e-3)')
    axes[1].set_xlabel('fraction of cells changed'); axes[1].set_ylabel('median log10 p after scrubbing')
    axes[0].legend(fontsize=6); axes[1].legend(fontsize=5)
    save(fig, d, 'stealing')


def fig_radio(S, d):
    r = S.get('radio', {}).get('by_method_rho', {})
    if not r:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for wm in ['gm_kgw', 'gm_gumbel']:
        pts = sorted((float(k.split('rho')[1]), np.median(v['read_lp'])) for k, v in r.items() if k.startswith(wm + '|'))
        r0 = [np.median(v['read_lp']) for k, v in r.items() if k.endswith('rho0')]
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker='o', ms=3, color=COL[wm], label=LAB[wm] + ' (reading)')
    ax.set_xlabel('fraction of watermarked training data'); ax.set_ylabel('median log10 p (student)')
    ax.legend(fontsize=6)
    save(fig, d, 'radioactivity')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    S = json.load(open(os.path.join(cfg['out_dir'], 'summary.json')))
    d = p(cfg.get('fig_dir', 'figures'))
    for f in (fig_qq,):
        f(cfg, S, d)
    for f in (fig_tpr_vs_pairs, fig_attacks, fig_survival, fig_pooled, fig_steal, fig_radio):
        try:
            f(S, d)
        except Exception as e:   # a missing section must not block the other figures
            print(f'{f.__name__} failed: {e}')


if __name__ == '__main__':
    main()
