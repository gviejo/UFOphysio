import numpy as np
import pickle
import os
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import *

data_dir = os.path.expanduser("~/Dropbox/UFOPhysio/data")

with open(os.path.join(data_dir, "ufo_reacs_ADN_DG.pkl"), "rb") as f:
    ufo_reacs_adn_dg = pickle.load(f)

with open(os.path.join(data_dir, "ufo_reacs_LMN_PSB.pkl"), "rb") as f:
    ufo_reacs_lmn_psb = pickle.load(f)

# (label, data_dict, color)
all_groups = [
    ('adn',  ufo_reacs_adn_dg,  'C0'),
    ('dg',   ufo_reacs_adn_dg,  'C1'),
    ('ca1',  ufo_reacs_adn_dg,  'C2'),
    ('hpc',  ufo_reacs_adn_dg,  'C3'),
    ('all',  ufo_reacs_adn_dg,  'C4'),
    ('lmn',  ufo_reacs_lmn_psb, 'C5'),
    ('psb',  ufo_reacs_lmn_psb, 'C6'),
]

n_cols = len(all_groups)
pdf_path = os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_reactivation_summary.pdf")

with PdfPages(pdf_path) as pdf:

    fig, ax = subplots(figsize=(6, 4))

    for g, data, color in all_groups:
        if g not in data:
            continue
        traces = []
        for s, tmp in data[g].items():
            # ax.plot(tmp, color=color, alpha=0.15, linewidth=0.6)
            traces.append(tmp.values.flatten())
        if traces:
            min_len = min(len(z) for z in traces)
            mean_trace = np.nanmean([z[:min_len] for z in traces], 0)
            t = tmp.index.values[:min_len]
            ax.plot(t, mean_trace, color=color, linewidth=2, label=g.upper())

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Time from UFO (s)", fontsize=9)
    ax.set_ylabel("Reactivation (z-score)", fontsize=9)
    ax.set_ylim(-5, 5)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)

    tight_layout()
    pdf.savefig(fig)
    close(fig)

# show()