"""Stage 7 figures from results/rev1/: strength-utility frontier at matched distortion (dataset-level evidence after
map-matching vs bigram and trip-length divergence, for GeoMark-KGW, its EOS-neutral variant, exact-hash KGW and the
post-hoc baseline at several rates). Writes figures/fig_frontier.pdf. Usage: python src/rev1/make_figs_rev1.py"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rev1.rcommon import OUT, ppath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.0,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
W2 = 7.16


def main():
    R = json.load(open(os.path.join(OUT, 'summary_rev1.json')))['posthoc']
    fam = [('GeoMark-KGW ($\\delta$=1, 2, 4)', ['a_g25_d1', 'gm_kgw', 'a_g25_d4'], OKABE[0], 'o', '-'),
           ('GeoMark-KGW, EOS-neutral ($\\delta$=2, 4)', ['gm_kgw_eosn', 'a_g25_d4_eosn'], OKABE[2], 's', '-'),
           ('exact-hash KGW ($\\delta$=1, 2)', ['exact_kgw_d1', 'exact_kgw'], OKABE[1], '^', '--'),
           ('post-hoc, rate 0.1--1', ['posthoc_r10', 'posthoc_r25', 'posthoc_r50', 'posthoc'], OKABE[3], 'D', ':')]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05))
    for lab, wms, c, m, ls in fam:
        for ax, (xk, xl) in zip(axes[:2], [('jsd_bigram', 'bigram JSD to real'), ('jsd_length_ovf', 'trip-length JSD to real')]):
            xs, ys = [], []
            for wm in wms:
                r = R.get(wm, {})
                u = r.get('utility') or {}
                p = r.get('mm')
                if u.get(xk) is None or not p:
                    continue
                xs.append(u[xk]); ys.append(min(300, p['median_mlp']))
            ax.plot(xs, ys, color=c, marker=m, ls=ls, ms=3.5, lw=1.0, label=lab)
            ax.set_xlabel(xl)

    # Stage 8: step-neutral default from results/rev2/summary_rev2.json (factorial cell coarse_bin_2r_sn)
    f2 = ppath('results', 'rev2', 'summary_rev2.json')
    if os.path.exists(f2):
        sn = json.load(open(f2)).get('factorial', {}).get('coarse_bin_2r_sn', {})
        u, p = sn.get('utility') or {}, (sn.get('mm') or {}).get('pooled')
        if p and u:
            for ax, xk in zip(axes, ('jsd_bigram', 'jsd_length_ovf')):
                ax.plot([u[xk]], [p['median_mlp']], color=OKABE[6], marker='h', ls='none', ms=4.5,
                        label='GeoMark-KGW, step-neutral ($\\delta$=2)' if ax is axes[0] else None)
    from matplotlib.ticker import NullFormatter, LogLocator
    for ax in axes:
        ax.set_yscale('log'); ax.set_xscale('log'); ax.axhline(3, color='k', lw=0.5, ls='-.')
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('$-\\log_{10}p$ after map-matching')
    axes[0].set_title('(a)', loc='left'); axes[1].set_title('(b)', loc='left')
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6, rect=(0, 0.2, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, fontsize=5.6)
    os.makedirs(ppath('figures'), exist_ok=True)
    fig.savefig(ppath('figures', 'fig_frontier.pdf'))
    print('wrote figures/fig_frontier.pdf')


if __name__ == '__main__':
    main()
