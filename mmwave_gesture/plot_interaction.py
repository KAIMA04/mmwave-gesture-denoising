"""Figure of the local-minus-global interaction contrast (per user, 95% bootstrap CI) on Pantomime.

Reads the test list written by stats.py and plots, for each protocol, the contrast of skipping DBSCAN
(raw vs. DBSCAN, and gate vs. gate->DBSCAN) under per-sample and fixed-scale normalisation.
Error bars: unadjusted 95% bootstrap CIs. Colour/shape: normalisation. Filled markers: Wilcoxon p < 0.05 after Holm within the test's family
(`p_wil_fam`; falls back to `p_wil_holm` for older stats files).

Usage: python -m mmwave_gesture.plot_interaction --stats stats.json --out interaction.pdf
"""
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [('LUO5', 'LUO5'), ('open->office', 'Open$\\to$Office'), ('office->open', 'Office$\\to$Open')]
SERIES = [('per-sample', 'C0', 'o', {'raw/DB': 'interaction', 'gate/G$\\to$DB': 'interaction gate-vs-gateDB'}),
          ('fixed scale', 'C1', 's', {'raw/DB': 'interaction FIXED raw-vs-DB', 'gate/G$\\to$DB': 'interaction FIXED gate-vs-gateDB'})]


def load(path):
    d = json.load(open(path))
    return d['tests'] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stats', default='stats.json')
    ap.add_argument('--out', default='interaction.pdf')
    ap.add_argument('--margin', type=float, default=2.0, help='equivalence margin drawn as a grey band (0: none)')
    a = ap.parse_args()
    rows = {(r['ds'], r['proto'], r['label']): r for r in load(a.stats)}
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.55), sharey=False)
    xs = list(SERIES[0][3])
    for ax, (proto, title) in zip(axes, PANELS):
        for k, (name, col, mk, labels) in enumerate(SERIES):
            for i, x in enumerate(xs):
                r = rows[('pant', proto, labels[x])]
                lo, hi = r['ci']
                xi = i + (k - 0.5) * 0.28
                ax.errorbar(xi, r['mean'], yerr=[[r['mean'] - lo], [hi - r['mean']]], fmt=mk, color=col, ms=3.5, lw=0.9, capsize=1.5,
                            mfc=col if r.get('p_wil_fam', r.get('p_wil_holm')) < 0.05 else 'white')
        ax.axhline(0, color='0.5', lw=0.6)
        if a.margin > 0:
            ax.axhspan(-a.margin, a.margin, color='0.85', lw=0, zorder=0.5)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=5.5)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_title(title, fontsize=7, pad=2)
        ax.tick_params(length=2, pad=1)
    axes[0].set_ylabel('local $-$ global (pp)')
    # legend: colour and shape give the scale; fill gives significance (open marker: family-wise Holm p >= 0.05)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls='', marker=mk, color=col, ms=3.5, label=name) for name, col, mk, _ in SERIES]
    handles.append(Line2D([], [], ls='', marker='o', color='0.4', mfc='white', ms=3.5, label='open: $p_H\\geq$0.05'))
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(lo, hi + 0.35 * (hi - lo))
    axes[1].legend(handles=handles, fontsize=5, frameon=False, loc='upper right', handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(a.out)


if __name__ == '__main__':
    main()
