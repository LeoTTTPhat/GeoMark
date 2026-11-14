"""Stage 5 paper figures -> figures/fig_*.pdf (vector). Sources: results/summary.json, results/stats_tests.json and the
raw files they were built from (calibration_ext/*.npz, keyvar/*.jsonl, eval/*/pooled.jsonl). Every figure supports one
claim in results/analysis.md. Sized for IEEE two-column pages (single column 3.5 in, double column 7.16 in), STIX
(Times-like) fonts at 7-8 pt, Okabe-Ito colour-blind-safe palette with a distinct marker per scheme (secondary encoding;
palette validated with the dataviz validator: all checks pass, CVD worst pair 7.6 -> markers are required).
Usage: python src/make_figures.py [--root DIR]"""
import os, sys, json, glob, argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W1, W2 = 3.5, 7.16
# fixed scheme -> (colour, marker, label); colour follows the scheme in every figure
SCH = {
    'gm_kgw':    ('#0072B2', 'o', 'GeoMark-KGW'),
    'gm_gumbel': ('#D55E00', 's', 'GeoMark-Gumbel-H'),
    'exact_kgw': ('#009E73', '^', 'Exact-hash KGW'),
    'posthoc':   ('#CC79A7', 'D', 'Post-hoc (same hash)'),
    'a_g25_d4':  ('#E69F00', 'v', r'GeoMark-KGW $\gamma$=.25 $\delta$=4'),
    'a_g50_d4':  ('#56B4E9', 'P', r'GeoMark-KGW $\gamma$=.5 $\delta$=4'),
    'wtrace':    ('#000000', 'x', 'W-Trace'),
    'gm_kgw_sn': ('#7a3fb0', 'h', 'GeoMark-KGW step-neutral'),
}
GREY, INK, MUTED = '#8c8c8c', '#1a1a1a', '#5c5c5c'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.titlesize': 7.5,
    'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6',
    'grid.linewidth': 0.5, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'lines.linewidth': 1.2, 'lines.markersize': 3.5,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.titlepad': 3,
})
ATK_LABEL = {'none': 'none', 'noise10': 'noise 10 m', 'noise30': 'noise 30 m', 'noise100': 'noise 100 m', 'down2': 'down x2',
             'down3': 'down x3', 'down5': 'down x5', 'retime15': 'retime 15 min', 'retime60': 'retime 60 min',
             'crop25': 'crop 25%', 'crop50': 'crop 50%', 'splice30': 'splice 30%', 'mm': 'map-match', 'noise30+mm': 'noise30+mm',
             'mmrs': 'mm + resample', 'regen': 'regenerate', 'mix10': 'mix 10%', 'mix30': 'mix 30%', 'mix50': 'mix 50%'}
GEN_LABEL = {'gpt_porto_r9_s0': 'Porto, small GPT', 'gpt_geolife_r9_s0': 'GeoLife, small GPT', 'qwen15_porto_r9': 'Porto, Qwen2.5-1.5B',
             'qwen15_geolife_r9': 'GeoLife, Qwen2.5-1.5B'}
WRITTEN = []


def save(fig, d, name):
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + '.pdf')
    fig.savefig(path)
    plt.close(fig)
    WRITTEN.append(path)
    print('wrote', path)


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return ph, max(0, c - h), min(1, c + h)


def panel_tag(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, fontweight='bold', color=INK)


# ---------------------------------------------------------------------------------------------------------- F1 calibration
def fig_calibration(root, S, d):
    """P1: randomised null p-values are uniform (QQ with 95% pointwise envelope, 1e5 real traces x random keys);
    key-conditional FPR over 200 keys is over-dispersed relative to Binomial(3000, mean)."""
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.15))
    for ax, ds, tag in zip(axes[:2], ['porto', 'geolife'], ['(a)', '(b)']):
        n = None
        for wm in ['gm_kgw', 'gm_gumbel', 'exact_kgw']:
            f = os.path.join(root, 'results', 'calibration_ext', f'{ds}_r9_{wm}_real.npz')
            if not os.path.exists(f):
                continue
            pv = np.sort(10 ** np.load(f)['lp_rand'])
            n = len(pv)
            e = (np.arange(1, n + 1) - 0.5) / n
            idx = np.unique(np.geomspace(1, n, 400).astype(int)) - 1
            c, m, lab = SCH[wm]
            ax.loglog(e[idx], pv[idx], color=c, lw=1.0, label=lab)
        if n:
            i = np.unique(np.geomspace(1, n, 300).astype(int))
            lo, hi = stats.beta.ppf(0.025, i, n - i + 1), stats.beta.ppf(0.975, i, n - i + 1)
            ax.fill_between((i - 0.5) / n, lo, hi, color=GREY, alpha=0.25, lw=0, label='95% envelope')
        ax.loglog([1e-5, 1], [1e-5, 1], color=MUTED, lw=0.6, ls='--')
        ks = {wm: S['calibration_ext'].get(f'{ds}_r9_{wm}_real', {}).get('ks_randomized', {}).get('ks_p') for wm in ['gm_kgw', 'gm_gumbel', 'exact_kgw']}
        ax.set_title(f"{'Porto' if ds == 'porto' else 'GeoLife'} r9, $10^5$ real-trace nulls", loc='left')
        ax.text(0.97, 0.04, 'KS p: ' + ', '.join(f'{v:.2f}' for v in ks.values() if v is not None), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6, color=MUTED)
        ax.set_xlabel('expected p (uniform)'); ax.set_xlim(1e-5, 1); ax.set_ylim(1e-5, 1)
        panel_tag(ax, tag)
    axes[0].set_ylabel('observed null p (randomized)')
    axes[0].legend(loc='upper left', handlelength=1.4)
    # (c) key-conditional FPR at 1e-3, Gumbel-H and KGW, Porto/GeoLife
    ax = axes[2]
    recs = [json.loads(l) for l in open(os.path.join(root, 'results', 'keyvar', 'null.jsonl'))]
    by = defaultdict(list)
    for r in recs:
        by[(r['cell'], r['scheme'])].append(r['k_le_m3'])
    bins = np.arange(-0.5, 17.5, 1)
    for (cell, wm), ls in [(('porto_r9', 'gm_gumbel'), '-'), (('geolife_r9', 'gm_gumbel'), ':')]:
        k = np.array(by[(cell, wm)])
        h, _ = np.histogram(k, bins)
        ax.step(np.arange(17), h, where='mid', color=SCH[wm][0], ls=ls, lw=1.1, label=f"Gumbel-H {'Porto' if cell.startswith('porto') else 'GeoLife'} (200 keys)")
        pm = stats.binom.pmf(np.arange(17), 3000, k.mean() / 3000) * len(k)
        ax.plot(np.arange(17), pm, color=MUTED, lw=0.7, ls=ls, marker='.', ms=2)
    kk = np.array(by[('porto_r9', 'gm_kgw')])
    h, _ = np.histogram(kk, bins)
    ax.step(np.arange(17), h, where='mid', color=SCH['gm_kgw'][0], lw=1.1, label='KGW Porto (conservative)')
    ax.plot([], [], color=MUTED, lw=0.7, marker='.', ms=2, label='Binomial(3000, mean FPR)')
    ax.axvline(3, color=INK, lw=0.6, ls='--'); ax.text(3.3, ax.get_ylim()[1] * 0.55, 'nominal', fontsize=5.6, color=INK, va='top')
    ax.set_xlabel('false positives at $p\\leq10^{-3}$ among 3,000 real traces')
    ax.set_ylabel('number of keys'); ax.set_title('Key-conditional FPR (single owner)', loc='left')
    ax.legend(loc='upper right', fontsize=5.2, handlelength=1.2, bbox_to_anchor=(1.0, 1.0))
    panel_tag(ax, '(c)')
    fig.tight_layout(w_pad=1.0)
    save(fig, d, 'fig_calibration')


# ------------------------------------------------------------------------------------------------ F2 TPR vs distinct pairs
def fig_tpr_vs_pairs(S, d):
    """Capacity: per-trace TPR rises with the number of distinct (seed, cell) pairs; Wilson 95% CIs per bin."""
    tv = S['tpr_vs_pairs']
    gens = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0']
    fig, axes = plt.subplots(2, 2, figsize=(W1, 3.3), sharex=True, sharey=True)
    for i, gen in enumerate(gens):
        for j, atk in enumerate(['none', 'mm']):
            ax = axes[i, j]
            for wm in ['exact_kgw', 'gm_kgw', 'a_g25_d4', 'a_g50_d4']:   # Stage 7: Gumbel-H dropped (duplicate-driven artifact)
                r = tv.get(f'{gen}|{wm}|{atk}')
                if not r:
                    continue
                xs, ys, lo, hi = [], [], [], []
                for b, v in r.items():
                    a, z = b.strip('[)').split(',')
                    a, z = float(a), float(z)
                    if v['n'] < 30:
                        continue
                    xs.append(a + (min(z, 80) - a) / 2)
                    p, l, h = wilson(round(v['tpr_1e3'] * v['n']), v['n'])
                    ys.append(p); lo.append(p - l); hi.append(h - p)
                c, m, lab = SCH[wm]
                ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=m, ms=3, lw=1.0, elinewidth=0.6, capsize=0, label=lab)
            ax.set_title(f"{GEN_LABEL[gen]}, {ATK_LABEL[atk]}", loc='left', fontsize=6.8)
            ax.set_ylim(-0.02, 1.02)
            if i == 1:
                ax.set_xlabel('distinct scored pairs (bin center)')
            if j == 0:
                ax.set_ylabel('TPR at FPR $10^{-3}$')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=5.4, handlelength=1.2, bbox_to_anchor=(0.5, -0.005))
    save(fig, d, 'fig_tpr_vs_pairs')


# --------------------------------------------------------------------------------------------------- F3 many-key per-trace
def fig_keyvar(root, S, d):
    """Key effect: per-key TPR over 50 keys (12 for Qwen) -- Gumbel-H spread exceeds the binomial band, KGW does not."""
    recs = [json.loads(l) for l in open(os.path.join(root, 'results', 'keyvar', 'tpr.jsonl'))]
    by = defaultdict(list)
    for r in recs:
        by[(r['gen'], r['wm'], r['attack'])].append(r['tpr_1e3'])
    gens = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9']
    atks = ['none', 'noise30', 'down3', 'crop50']
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, gen in zip(axes, gens):
        for j, atk in enumerate(atks):
            for off, wm in [(-0.18, 'gm_kgw'), (0.18, 'gm_gumbel')]:
                v = np.array(by.get((gen, wm, atk), []))
                if not len(v):
                    continue
                c, m, lab = SCH[wm]
                x = j + off + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(x, v, s=5, color=c, marker=m, alpha=0.7, lw=0, label=lab if j == 0 else None)
                mu = v.mean(); sb = np.sqrt(mu * (1 - mu) / 200)
                ax.plot([j + off - 0.13, j + off + 0.13], [mu, mu], color=INK, lw=1.0)
                ax.fill_between([j + off - 0.13, j + off + 0.13], mu - 2 * sb, mu + 2 * sb, color=GREY, alpha=0.35, lw=0)
        e = S['stats_key_effect'].get(f'{gen}|gm_gumbel|none') if 'stats_key_effect' in S else None
        ax.set_xticks(range(len(atks))); ax.set_xticklabels([ATK_LABEL[a] for a in atks], rotation=0)
        nk = len(by.get((gen, 'gm_gumbel', 'none'), []))
        ax.set_title(f"{GEN_LABEL[gen]} ({nk} keys x 200 traces)", loc='left')
        if e:
            ax.text(0.98, 0.95, f"Gumbel-H key effect: var ratio {e['var_ratio']:.1f}, $\\chi^2$ p={e['p']:.0e}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=5.8, color=MUTED)
    axes[0].set_ylabel('per-key TPR at FPR $10^{-3}$')
    axes[0].plot([], [], color=INK, lw=1.0, label='mean over keys')
    axes[0].fill_between([], [], [], color=GREY, alpha=0.35, label='binomial $\\pm2$ sd')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=5.6, handlelength=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, d, 'fig_keyvar')


# ------------------------------------------------------------------------------------------------------- F4 dataset level
def fig_pooled(root, S, d):
    """Headline: dataset-level (pooled, de-duplicated) test; median and IQR of -log10 p vs release size."""
    gens = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']   # Stage 7: fourth panel
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), sharey=True)
    for ax, gen in zip(axes, gens):
        acc = defaultdict(list)
        for l in open(os.path.join(root, 'results', 'eval', gen, 'pooled.jsonl')):
            r = json.loads(l)
            if r['T'] is None:
                acc[(r['rid'].split('__')[0], r['attack'], r['m'])].extend(r['lps'])
        for wm in ['exact_kgw', 'gm_kgw', 'gm_kgw_sn', 'gm_gumbel', 'a_g25_d4']:
            for atk, ls in [('none', '-'), ('mm', '--'), ('mmrs', ':')]:
                ms = sorted(m for (w, a, m) in acc if w == wm and a == atk)
                if not ms:
                    continue
                med, q1, q3 = [], [], []
                for m in ms:
                    v = -np.clip(np.asarray(acc[(wm, atk, m)]), -300, 0)
                    med.append(np.median(v)); q1.append(np.quantile(v, .25)); q3.append(np.quantile(v, .75))
                c, mk, lab = SCH[wm]
                ax.plot(ms, med, color=c, marker=mk, ls=ls, ms=3, label=f'{lab}' if atk == 'none' else None)
                ax.fill_between(ms, q1, q3, color=c, alpha=0.12, lw=0)
        ax.axhline(3, color=INK, lw=0.6, ls='-.')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.1, 400)
        ax.set_xlabel('traces in released dataset'); ax.set_title(GEN_LABEL[gen], loc='left')
    axes[0].set_ylabel('$-\\log_{10}$ p (pooled test)')
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=MUTED, ls=s) for s in ['-', '--', ':']] + [plt.Line2D([], [], color=INK, ls='-.', lw=0.6)]
    l += ['no attack', 'map-matched', 'map-matched + resampled', '$p=10^{-3}$ (values clipped at 300)']
    fig.tight_layout(w_pad=0.6, rect=(0, 0.17, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=5.8, handlelength=2.0, bbox_to_anchor=(0.5, -0.02))
    save(fig, d, 'fig_pooled')


# ------------------------------------------------------------------------------------------------------- F5 attack heatmap
def fig_attacks(S, d):
    """Robustness: per-trace TPR (left) and dataset-level median -log10 p for 100-trace releases (right), Porto small GPT."""
    det, po = S['detection'], S['pooled']
    gen = 'gpt_porto_r9_s0'
    atks = ['none', 'noise10', 'noise30', 'noise100', 'down2', 'down3', 'down5', 'crop25', 'crop50', 'splice30',
            'retime15', 'retime60', 'mm', 'noise30+mm', 'mmrs', 'regen']
    wms = ['exact_kgw', 'posthoc', 'gm_kgw', 'gm_gumbel', 'a_g25_d4', 'a_g50_d4']
    A = np.full((len(atks), len(wms)), np.nan); B = np.full_like(A, np.nan)
    for i, a in enumerate(atks):
        for j, w in enumerate(wms):
            c = det.get(f'{gen}|{w}|{a}')
            if c:
                A[i, j] = c['tpr_1e3'][0]
            p = po.get(f'{gen}|{w}|{a}|-|m100')
            if p:
                B[i, j] = min(300.0, -p['median_lp']) if np.isfinite(p['median_lp']) else 300.0
    cmap = LinearSegmentedColormap.from_list('blues', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.3))
    for ax, M, title, fmt, norm in [(axes[0], A, 'per-trace TPR at FPR $10^{-3}$', '{:.2f}', matplotlib.colors.Normalize(0, 1)),
                                     (axes[1], B, 'dataset level: median $-\\log_{10}$ p, 100-trace releases', '{:.0f}',
                                      matplotlib.colors.LogNorm(1, 300))]:
        Mm = np.ma.masked_invalid(M)
        im = ax.imshow(np.where(np.isnan(M), np.nan, np.clip(M, 1e-3 if ax is axes[0] else 1, None)), cmap=cmap, norm=norm, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    ax.text(j, i, 'n/a', ha='center', va='center', fontsize=5.2, color=MUTED); continue
                v = M[i, j]
                dark = norm(max(v, 1e-3 if ax is axes[0] else 1)) > 0.55
                s = fmt.format(v) if not (ax is axes[1] and v >= 300) else '>300'
                ax.text(j, i, s, ha='center', va='center', fontsize=5.6, color='white' if dark else INK)
        ax.set_xticks(range(len(wms)))
        ax.set_xticklabels(['Exact-hash\nKGW', 'Post-hoc\n(same hash)', 'GeoMark\nKGW', 'GeoMark\nGumbel-H*', 'KGW $\\delta$=4\n$\\gamma$=.25',
                            'KGW $\\delta$=4\n$\\gamma$=.5'], fontsize=5.8)
        ax.set_yticks(range(len(atks))); ax.set_yticklabels([ATK_LABEL[a] for a in atks])
        ax.grid(False); ax.set_title(title, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_visible(False)
    axes[1].set_yticklabels([])
    fig.text(0.01, -0.01, '*Gumbel-H per-trace values are 3-key estimates (upper-tail keys); the 50-key mean is 0.103 (Fig. keyvar). '
             'Map-matching rows use 3 seed-0 releases x 1,000 traces; others 9 releases.', fontsize=5.6, color=MUTED)
    fig.tight_layout(w_pad=0.4)
    save(fig, d, 'fig_attacks')


# ------------------------------------------------------------------------------------------------------- F6 resolution
def fig_resolution(S, d):
    """Resolution trade-off r8/r9/r10 under map-matching (key-level 95% CIs, 3 keys)."""
    det = S['detection']
    atks = ['none', 'noise30', 'mm', 'noise30+mm', 'mmrs']
    cols = {8: '#009E73', 9: '#0072B2', 10: '#D55E00'}
    mks = {8: '^', 9: 'o', 10: 's'}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0))
    for ax, wm in zip(axes[:2], ['gm_kgw', 'gm_gumbel']):
        for k, r in enumerate([8, 9, 10]):
            gen = f'gpt_porto_r{r}_s0'
            y, lo, hi = [], [], []
            for a in atks:
                c = det.get(f'{gen}|{wm}|{a}')
                v = c.get('tpr_1e3_keylevel', c['tpr_1e3']) if c else [np.nan] * 3
                y.append(v[0]); lo.append(v[0] - max(0, v[1])); hi.append(v[2] - v[0])
            x = np.arange(len(atks)) + (k - 1) * 0.18
            ax.errorbar(x, y, yerr=[lo, hi], color=cols[r], marker=mks[r], ls='none', ms=3.5, elinewidth=0.7, label=f'r{r}')
        ax.set_xticks(range(len(atks))); ax.set_xticklabels([ATK_LABEL[a] for a in atks], rotation=25, ha='right')
        ax.set_title(f'{SCH[wm][2]}, Porto small GPT', loc='left'); ax.set_ylim(bottom=0)
    axes[0].set_ylabel('TPR at FPR $10^{-3}$ (3-key mean, 95% CI)')
    axes[0].legend(loc='upper right', handlelength=1)
    ax = axes[2]
    for k, r in enumerate([8, 9, 10]):
        gen = f'gpt_porto_r{r}_s0'
        y = [det.get(f'{gen}|gm_kgw|{a}', {}).get('median_n_f', [np.nan])[0] for a in atks]
        s = [det.get(f'{gen}|gm_kgw|{a}', {}).get('surv_f', [np.nan])[0] for a in atks]
        ax.plot(np.arange(len(atks)), s, color=cols[r], marker=mks[r], ms=3.5, label=f'r{r} (median pairs, no attack: {y[0]:.0f})')
    ax.set_xticks(range(len(atks))); ax.set_xticklabels([ATK_LABEL[a] for a in atks], rotation=25, ha='right')
    ax.set_ylabel('surviving fine pairs (fraction)'); ax.set_title('Fine-pair survival (GeoMark-KGW)', loc='left')
    ax.legend(loc='lower left', fontsize=5.6, handlelength=1)
    fig.tight_layout(w_pad=0.6)
    save(fig, d, 'fig_resolution')


# ------------------------------------------------------------------------------------------------------- F7 Lemma 3
def fig_lemma3(root, S, R, d):
    """P2: (a) Lemma-3 predicted vs measured surviving fraction under Gaussian noise; (b) closed-form vs Monte Carlo
    cell-crossing probability from H3 geometry."""
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.45, 2.1))
    ax = axes[0]
    rows = R['lemma3'].get('rows', [])
    groups = {'KGW, k<=1': (SCH['gm_kgw'][0], 'o'), 'KGW, k>=2': ('#56B4E9', 'P'), 'Gumbel-H (masking)': (SCH['gm_gumbel'][0], 's'),
              'Post-hoc': (SCH['posthoc'][0], 'D')}
    for r in rows:
        wm = r['wm']
        if 'gum' in wm:
            g = 'Gumbel-H (masking)'
        elif wm == 'posthoc':
            g = 'Post-hoc'
        elif wm in ('gm_kgw_k2', 'a_k3'):
            g = 'KGW, k>=2'
        else:
            g = 'KGW, k<=1'
        c, m = groups[g]
        ax.scatter(r['pred_f'], r['meas_f'], s=9, color=c, marker=m, lw=0, alpha=0.85)
    for g, (c, m) in groups.items():
        ax.scatter([], [], s=9, color=c, marker=m, label=g)
    ax.plot([0.3, 1], [0.3, 1], color=MUTED, lw=0.6, ls='--')
    ax.set_xlabel('approximation (Eq. 7)'); ax.set_ylabel('measured surviving fine pairs')
    ax.set_title('Noise 10/30/100 m, all generators', loc='left'); ax.legend(loc='upper left', fontsize=5.5, handlelength=1)
    panel_tag(ax, '(a)')
    ax = axes[1]
    geo = json.load(open(os.path.join(root, 'results', 'geometry_mc.json')))
    rc = {6: '#999999', 7: '#009E73', 8: '#0072B2', 9: '#D55E00', 10: '#CC79A7'}
    for g in geo:
        if g['kind'] != 'gauss_sigma':
            continue
        ax.scatter(g['closed_form'], g['empirical'], s=10, color=rc.get(g['res'], INK), marker='o', lw=0)
    for r, c in rc.items():
        if any(g['res'] == r for g in geo):
            ax.scatter([], [], s=10, color=c, label=f'r{r}')
    ax.plot([1e-3, 1], [1e-3, 1], color=MUTED, lw=0.6, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('closed form 0.921$\\sigma$/a'); ax.set_ylabel('Monte Carlo crossing prob.')
    ax.set_title('Cell crossing, Gaussian $\\sigma$', loc='left'); ax.legend(loc='upper left', fontsize=5.5, handlelength=1)
    panel_tag(ax, '(b)')
    fig.tight_layout(w_pad=0.8)
    save(fig, d, 'fig_lemma3')


# --------------------------------------------------------------------------------------------- F8 strength-utility trade-off
def fig_tradeoff(S, d):
    """Operating point: per-trace TPR (none and map-matched) vs bigram JSD to real data (Porto small GPT)."""
    det, U = S['detection'], S['utility']['gpt_porto_r9_s0']
    gen = 'gpt_porto_r9_s0'
    pts = [('exact_kgw', 'Exact-hash'), ('posthoc', 'Post-hoc'), ('a_g25_d1', '.25/1'), ('gm_kgw', '.25/2'), ('a_g25_d4', '.25/4'),
           ('a_g50_d1', '.5/1'), ('a_g50_d2', '.5/2'), ('a_g50_d4', '.5/4'), ('gm_gumbel', 'Gumbel-H')]
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.2), sharey=True)
    for ax, atk in zip(axes, ['none', 'mm']):
        for wm, lab in pts:
            c = det.get(f'{gen}|{wm}|{atk}'); u = U.get(wm)
            if not c or not u:
                continue
            col, mk = (SCH.get(wm, (SCH['gm_kgw'][0], 'o'))[:2]) if wm in SCH else (('#0072B2', 'o') if wm.startswith('a_g25') else ('#56B4E9', 'P'))
            x, xl, xh = u['jsd_bigram']
            y = c['tpr_1e3'][0]
            ax.errorbar(y, x, xerr=None, yerr=[[x - xl], [xh - x]] if np.isfinite(xl) else None, color=col, marker=mk, ms=4, elinewidth=0.6)
            off = {'a_g25_d1': (4, -8), 'a_g50_d1': (4, 3), 'gm_kgw': (4, -4), 'a_g50_d2': (4, 1)}.get(wm, (3, 2))
            ax.annotate(lab, (y, x), xytext=off, textcoords='offset points', fontsize=5.6, color=INK)
        km = S['utility']['keymix'].get('gpt_porto_r9_s0|gm_gumbel|keymix')
        kv = S['keyvar']['tpr'].get(f'{gen}|gm_gumbel|{atk}')
        if km and kv:
            ax.scatter(kv['tpr_1e3'][0], km['jsd_bigram'][0], marker='*', s=30, color=SCH['gm_gumbel'][0], zorder=3)
            ax.annotate('Gumbel-H, 50-key mix', (kv['tpr_1e3'][0], km['jsd_bigram'][0]), xytext=(3, -7), textcoords='offset points', fontsize=5.6)
        nv = U['none']['jsd_bigram'][0]; rv = U['real_val']['jsd_bigram']; rv = rv[0] if isinstance(rv, list) else rv
        ax.axhline(nv, color=MUTED, lw=0.6, ls='--'); ax.axhline(rv, color=GREY, lw=0.6, ls=':')
        ax.text(0.98, nv, 'no watermark', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=5.6, color=MUTED)
        ax.text(0.98, rv, 'real vs real', transform=ax.get_yaxis_transform(), ha='right', va='top', fontsize=5.6, color=MUTED)
        ax.set_xlabel(f'per-trace TPR at FPR $10^{{-3}}$ ({ATK_LABEL[atk]})')
        ax.set_title('Porto small GPT, labels $\\gamma/\\delta$', loc='left')
    axes[0].set_ylabel('bigram JSD to real (lower = better)')
    fig.tight_layout(w_pad=0.6)
    save(fig, d, 'fig_tradeoff')


# ------------------------------------------------------------------------------------------------------- F9 ablations
def fig_ablation(S, d):
    """Ablations on Porto small GPT: dataset-level median -log10 p (100-trace releases) per scheme variant and attack."""
    po = S['pooled']
    gen = 'gpt_porto_r9_s0'
    var = [('a_k0', 'k=0'), ('gm_kgw', 'k=1 (default)'), ('gm_kgw_k2', 'k=2'), ('a_k3', 'k=3'), ('a_bin0', 'no time bin'),
           ('a_bin120', '120-min bins'), ('a_ctxmid', 'ctx r-1'), ('a_1r', 'single resolution'), ('exact_kgw', 'exact hash')]
    atks = [('none', 'o'), ('noise30', 's'), ('mm', '^'), ('mmrs', 'v'), ('retime60', 'x')]
    cols = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#000000']
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for j, ((a, m), c) in enumerate(zip(atks, cols)):
        y = []
        for wm, _ in var:
            p = po.get(f'{gen}|{wm}|{a}|-|m100')
            y.append(np.nan if not p else (300 if not np.isfinite(p['median_lp']) else -p['median_lp']))
        ax.plot(np.arange(len(var)) + (j - 2) * 0.1, y, ls='none', marker=m, color=c, ms=3.5, label=ATK_LABEL[a])
    ax.axhline(3, color=INK, lw=0.6, ls='-.')
    ax.set_yscale('log'); ax.set_xticks(range(len(var))); ax.set_xticklabels([l for _, l in var], rotation=35, ha='right')
    ax.set_ylabel('median $-\\log_{10}$ p (100-trace pooled)'); ax.set_title('KGW ablations ($\\gamma$=.25, $\\delta$=2), Porto small GPT', loc='left')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=5.6, handlelength=1, columnspacing=0.8)
    ax.set_title('KGW ablations ($\\gamma$=.25, $\\delta$=2), Porto small GPT', loc='left', pad=16)
    fig.tight_layout()
    save(fig, d, 'fig_ablation')


# ------------------------------------------------------------------------------------------------------- F10 bin shift
def fig_shift(S, d):
    """Re-timestamping: bin-shift search radius J recovers retime60 at J>=2, at a small cost without attack."""
    sh = S['shift']
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.9), sharey=True)
    for ax, gen in zip(axes, ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0']):
        for atk, ls, m in [('none', '-', 'o'), ('retime15', '--', 's'), ('retime60', ':', '^')]:
            y = [sh.get(f'{gen}|a_g25_d4|{atk}|J{j}', {}).get('tpr_1e3', [np.nan])[0] for j in range(4)]
            ax.plot(range(4), y, color=SCH['a_g25_d4'][0], ls=ls, marker=m, ms=3, label=ATK_LABEL[atk])
        ax.set_xticks(range(4)); ax.set_xlabel('bin-shift radius J'); ax.set_title(GEN_LABEL[gen], loc='left'); ax.set_ylim(-0.02, 1)
    axes[0].set_ylabel('TPR at FPR $10^{-3}$ ($\\gamma$=.25, $\\delta$=4)')
    axes[0].legend(loc='lower right', fontsize=5.6, handlelength=1.6)
    fig.tight_layout(w_pad=0.5)
    save(fig, d, 'fig_shift')


# ------------------------------------------------------------------------------------------------------- F11 stealing
def fig_stealing(R, d):
    """P5: spoofing per trace (Wilson CI, 5,000 attempts) vs dataset-level spoofing (fraction of forged 100-trace pools
    with p<=1e-3), and informed vs random scrubbing at N=1e5."""
    sp = R['spoofing']
    Ns = [1000, 10000, 100000]
    corp = [('k1', 'k=1', '#0072B2', 'o'), ('k2', 'k=2', '#009E73', '^'), ('rot4', '4-key rotation', '#D55E00', 's')]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0))
    for c, lab, col, m in corp:
        keys = ['key0', 'key1', 'key2', 'key3'] if c == 'rot4' else ['key0']
        y, lo, hi, pf = [], [], [], []
        for N in Ns:
            vals = [sp[f'{c}|N{N}|{k}'] for k in keys if f'{c}|N{N}|{k}' in sp]
            kk = sum(v['k'] for v in vals); nn = sum(v['n'] for v in vals)
            p, l, h = wilson(kk, nn)
            y.append(p); lo.append(p - l); hi.append(h - p)
            pf.append(np.mean([v['pooled_frac_le_m3'] for v in vals]))
        axes[0].errorbar(Ns, y, yerr=[lo, hi], color=col, marker=m, ms=3.5, elinewidth=0.6, label=lab)
        axes[1].plot(Ns, pf, color=col, marker=m, ms=3.5, label=lab)
    rb = sp['k1|real_unmodified']['per_trace']
    axes[0].axhspan(rb[1], rb[2], color=GREY, alpha=0.3, lw=0, label='real traces (no forgery)')
    axes[0].set_xscale('log'); axes[0].set_xlabel('watermarked traces observed N'); axes[0].set_ylabel('per-trace spoof success ($p\\leq10^{-3}$)')
    axes[0].set_title('Per-trace forgery', loc='left'); axes[0].legend(loc='upper left', fontsize=5.6, handlelength=1.2)
    axes[1].set_xscale('log'); axes[1].set_xlabel('watermarked traces observed N'); axes[1].set_ylabel('forged 100-trace pools with $p\\leq10^{-3}$')
    axes[1].set_ylim(-0.03, 1.03); axes[1].set_title('Dataset-level forgery', loc='left')
    ax = axes[2]
    for c, lab, col, m in corp:
        xs = [sp[f'{c}|N100000|scrub_r{r}']['frac_changed'] for r in (0.25, 0.5, 1.0)]
        ys = [-sp[f'{c}|N100000|scrub_r{r}']['pooled_median'] for r in (0.25, 0.5, 1.0)]
        ax.plot(xs, ys, color=col, marker=m, ms=3.5, label=f'{lab}, informed')
        xr = [sp[f'{c}|random_scrub_r{r}']['frac_changed'] for r in (0.25, 0.5, 1.0)]
        yr = [-sp[f'{c}|random_scrub_r{r}']['pooled_median'] for r in (0.25, 0.5, 1.0)]
        ax.plot(xr, yr, color=col, marker=m, ms=3, ls=':', mfc='white', label='random recoloring' if c == 'k1' else None)
    ax.axhline(3, color=INK, lw=0.6, ls='-.')
    ax.set_yscale('log'); ax.set_xlabel('fraction of cells changed'); ax.set_ylabel('median $-\\log_{10}$ p (100-trace pools)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([1, 3, 10, 30]); ax.set_ylim(1, 50)
    ax.set_title('Scrubbing (estimates from N=$10^5$)', loc='left'); ax.legend(loc='lower left', fontsize=5.0, handlelength=1.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, d, 'fig_stealing')


# ------------------------------------------------------------------------------------------------------- F12 radioactivity
def fig_radio(S, R, d):
    """P4: student models trained on a fraction rho of watermarked data; teacher-key log10 p (mean, t-CI over 3 seeds)
    and key-randomised rank p (500 null keys; resolution 1/501)."""
    kn = S['radio']['keynull_by_rho']
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.45, 2.0))
    for wm in ['gm_kgw', 'gm_gumbel']:
        for mode, ls in [('sample', '-'), ('read', '--')]:
            rhos, y, lo, hi, rp = [], [], [], [], []
            for rho in ['0.0', '0.05', '0.2', '0.5', '1.0']:
                v = kn.get(f'keynull|{wm}|rho{rho}|{mode}')
                if not v:
                    continue
                rhos.append(float(rho)); m, l, h = v['lp_true']
                y.append(-m); lo.append(-m - (-h)); hi.append(-l - (-m))
                rp.append(R['radio'][f'keynull|{wm}|rho{rho}|{mode}']['max_rank_p'])
            c, mk, lab = SCH[wm]
            x = np.array(rhos) + (0.006 if mode == 'read' else 0)
            yl = np.clip(np.array(y) - np.clip(lo, 0, None), 0.03, None)
            axes[0].errorbar(x, y, yerr=[np.array(y) - yl, np.clip(hi, 0, None)], color=c, marker=mk, ls=ls, ms=3, elinewidth=0.6,
                             label=f"{lab.replace('GeoMark-', '')}, {'sampling' if mode == 'sample' else 'reading'}")
            axes[1].plot(x, rp, color=c, marker=mk, ls=ls, ms=3)
    axes[0].set_yscale('log'); axes[0].set_ylim(0.03, 100); axes[0].set_xlabel('watermarked fraction of student training data $\\rho$')
    axes[0].set_ylabel('$-\\log_{10}$ p, teacher key'); axes[0].set_title('Parametric test (mean, 3 students)', loc='left')
    axes[0].legend(loc='upper left', fontsize=5.4, handlelength=1.6)
    axes[1].axhline(0.05, color=INK, lw=0.6, ls='-.'); axes[1].axhline(1 / 501, color=GREY, lw=0.6, ls=':')
    axes[1].text(0.02, 1 / 501 * 1.2, 'resolution 1/501', fontsize=5.6, color=MUTED)
    axes[1].set_yscale('log'); axes[1].set_xlabel('$\\rho$'); axes[1].set_ylabel('largest rank p over 3 students')
    axes[1].set_title('Key-randomised rank test', loc='left')
    fig.tight_layout(w_pad=0.8)
    save(fig, d, 'fig_radio')


# ------------------------------------------------------------------------------------------------------- F13 cost
def fig_cost(R, d):
    """Cost: generation seconds per 1,000 traces (median over releases with >= 2,000 traces) by scheme and generator."""
    co = R['cost']
    gens = ['gpt_porto_r9_s0', 'gpt_geolife_r9_s0', 'qwen15_porto_r9', 'qwen15_geolife_r9']
    wms = ['none', 'exact_kgw', 'gm_kgw', 'gm_gumbel']
    cols = {'none': GREY, **{w: SCH[w][0] for w in wms[1:]}}
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    w = 0.2
    for j, wm in enumerate(wms):
        y = [co.get(f'{g}|{wm}', {}).get('sec_per_1k_median', np.nan) for g in gens]
        ax.bar(np.arange(len(gens)) + (j - 1.5) * w, y, w * 0.9, color=cols[wm], label='no watermark' if wm == 'none' else SCH[wm][2])
    ax.set_yscale('log'); ax.set_xticks(range(len(gens))); ax.set_xticklabels([GEN_LABEL[g].replace(', ', '\n') for g in gens])
    ax.set_ylabel('generation s per 1,000 traces')
    thr = R['calibration']['summary'].get('detect_tests_per_sec_median')
    ax.set_title(f"Generation cost (shared M3 Ultra); detection {thr:,.0f} traces/s" if thr else 'Generation cost', loc='left')
    ax.set_ylim(1, 1000)
    ax.legend(loc='upper left', fontsize=5.6, ncol=2, handlelength=1)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save(fig, d, 'fig_cost')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.root, 'results', 'summary.json')))
    R = json.load(open(os.path.join(a.root, 'results', 'stats_tests.json')))
    S['stats_key_effect'] = R['key_effect']['homogeneity']
    d = os.path.join(a.root, 'figures')
    jobs = [lambda: fig_calibration(a.root, S, d), lambda: fig_tpr_vs_pairs(S, d), lambda: fig_keyvar(a.root, S, d),
            lambda: fig_pooled(a.root, S, d), lambda: fig_attacks(S, d), lambda: fig_resolution(S, d),
            lambda: fig_lemma3(a.root, S, R, d), lambda: fig_tradeoff(S, d), lambda: fig_ablation(S, d), lambda: fig_shift(S, d),
            lambda: fig_stealing(R, d), lambda: fig_radio(S, R, d), lambda: fig_cost(R, d)]
    failed = 0
    for j in jobs:
        try:
            j()
        except Exception as e:
            import traceback; traceback.print_exc(); failed += 1
    print(f'{len(WRITTEN)} figures written, {failed} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
