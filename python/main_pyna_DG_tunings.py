import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from python.functions.functions import load_mean_waveforms
from detection import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans
from umap import UMAP

TUNING_CURVE_SMOOTH_SIGMA = 1
CIRCULAR_SMOOTH_N_TILES = 5


def smooth_circular_tuning_curves(data, sigma=TUNING_CURVE_SMOOTH_SIGMA, n_tiles=CIRCULAR_SMOOTH_N_TILES):
    """Gaussian-smooth 2D tuning curves (n_neurons, n_bins) along a circular bin axis."""
    n_bins = data.shape[1]
    stacked = np.tile(data, n_tiles)
    smoothed = gaussian_filter1d(stacked, sigma=sigma, axis=1)
    mid = n_tiles // 2
    return smoothed[:, mid * n_bins:(mid + 1) * n_bins]


###############################################################################################
# GENERAL infos
###############################################################################################
if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_DG.list'), delimiter = '\n', dtype = str, comments = '#')

channels_separation = np.genfromtxt(os.path.join(data_directory, 'channels_CA1_DG.txt'), delimiter =' ', dtype = str, comments ='#')
channels_separation = {a[0].split("/")[-1] : a[1].astype('int') for a in channels_separation}

tuning_curves = {
}
manifolds = {}
maxchs = []
ch_pos = {}

for s in datasets:
# for s in ['ADN-HPC/B5100/B5108/B5108-260330']:
    print(s)
    ###############################################################################################
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)

    basename = os.path.basename(path)
    filepath = os.path.join(path, "kilosort4", basename + ".nwb")
    if os.path.exists(filepath):
        nwb = nap.load_file(filepath)
        spikes = nwb['units']
        position = []
        columns = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        for k in columns:
            position.append(nwb[k].values)
        position = np.array(position)
        position = np.transpose(position)
        position = nap.TsdFrame(
            t=nwb['x'].t,
            d=position,
            columns=columns,
            time_support=nwb['position_time_support'])

        epochs = nwb['epochs']
        wake_ep = epochs[epochs.tags == "wake"]

        sws_ep = nwb['sws']
        # rem_ep = nwb['rem']

        # Waveform classification
        meanwavef, maxch = load_mean_waveforms(os.path.join(path, "kilosort4"))
        maxchs.append(maxch)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])

        nwb.close()

    else:

        data = ntm.load_session(path, 'neurosuite')
        spikes = data.spikes
        position = data.position
        wake_ep = data.epochs['wake']
        sws_ep = data.read_neuroscope_intervals('sws')
        #rem_ep = data.read_neuroscope_intervals('rem')
        # Waveform classification
        meanwavef, maxch = load_mean_waveforms(path)
        maxchs.append(maxch)
        spikes.maxch = np.hstack([chs for chs in maxch.values()])


    spikes.location = [v.lower() for v in spikes.location.values]
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)


    # Separate CA1 and DG neurons based on maxch
    location = spikes.location.copy()
    for i in location.index:
            if spikes.location[i] in ["hpc", "ca1"]:
                if spikes.maxch[i] < channels_separation[basename]:
                    location[i] = "ca1"
                else:
                    location[i] = "dg"
    spikes.location = location

    # Load cell type classification (interneuron / mossy / granule / other) written by
    # main_pyna_DG_classify_cells.py
    classification_path = os.path.join(path, "cell_classification.csv")
    if os.path.exists(classification_path):
        cell_class = pd.read_csv(classification_path, sep="\t").set_index("neuron")["cell_type"]
        spikes.cell_type = cell_class.reindex(spikes.index).fillna("other")
    else:
        spikes.cell_type = pd.Series("other", index=spikes.index)

    # maxch_values = spikes.maxch[location == "hpc"].values.reshape(-1, 1)
    # idx = np.where(location == "hpc")[0]
    # if len(maxch_values) > 0:
    #     clu = KMeans(n_clusters=2, random_state=0).fit(maxch_values)
    #     map_ = {np.argmin(clu.cluster_centers_.flatten()): "ca1", np.argmax(clu.cluster_centers_.flatten()): "dg"}
    #     for i, label in enumerate(clu.labels_):
    #         location[idx[i]] = map_[label]
    #     # location[(spikes.maxch<30) & (spikes.location == "hpc")] = "ca1"
    #     # location[(spikes.maxch>=30) & (spikes.location == "hpc")] = "dg"
    #     spikes.location = location
    # groups = spikes.metadata[['location', 'maxch']].groupby("location").groups
    # ch_pos[s] = {loc: spikes.maxch[groups[loc]].values for loc in groups.keys()}

    # ahv = nap.Tsd(position.t, np.unwrap(position['ry'])).derivative()
    ahv = np.unwrap(position['ry']).smooth(0.05).derivative()
    # if "B5108" in s:
    #     new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)
    # Clipping ahv values with tanh
    ahv_bound = 4*np.pi
    ahv = ahv_bound*np.tanh(ahv/ahv_bound)

    absahv = np.log10(np.abs(ahv)+1e-6)

    # Converting ahv to log(ahv)
    logahv = absahv.copy()
    logahv[ahv<0] = -logahv[ahv<0]

    # new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)
    new_wake_ep = position.time_support
    fast_wake_ep = position.time_support.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)
    # new_wake_ep = position.time_support.set_diff(np.abs(ahv).smooth(0.1).threshold(0.05, "below").time_support.drop_short_intervals(10.0))


    tmp = {}
    for g in meanwavef.keys():
        for i in range(meanwavef[g].shape[0]):
            n = spikes.index[spikes.group == g][i]
            tmp[n] = meanwavef[g][i]

    spikes = spikes[spikes.group == 0]

    spikes = spikes[spikes.rate > 1.0]

    pitch = position['rx']
    pitch[pitch>np.pi] = pitch[pitch>np.pi] - 2*np.pi
    pitch  = pitch * -1
    pitch = pitch.smooth(0.05)
    # pitch = np.log(pitch)
    # pitch[np.isnan(pitch)] = 0
    roll = position['rz']
    roll[roll>np.pi] = roll[roll>np.pi] - 2*np.pi

    pitch_roll = nap.TsdFrame(t=position.t, d=np.vstack([pitch.values, roll.values]).T, columns=['pitch', 'roll'], time_support=position.time_support)

    xz_vel = position[['x', 'z']].bin_average(0.05).derivative()
    lin_speed = nap.Tsd(t=xz_vel.t, d=np.hypot(xz_vel['x'].values, xz_vel['z'].values), time_support=xz_vel.time_support)
    lin_speed = lin_speed.smooth(0.1)
    lin_speed = np.log10(lin_speed+1e-6)
    lin_speed = np.clip(lin_speed, -3.5, -0.5)
    ###############################################################################################
    # TUNING CURVES
    ###############################################################################################

    tuning_curves[s] = {
        "hd" : nap.compute_tuning_curves(
            spikes, position['ry'], bins=61, range=(0, 2 * np.pi), feature_names=['angle'], epochs=new_wake_ep
        ),
        "ahv" : nap.compute_tuning_curves(
            spikes, logahv, bins=20, range = (-4.0, 4.0), feature_names=['ahv'], epochs=new_wake_ep
        ),
        "absahv" : nap.compute_tuning_curves(
            spikes, absahv, bins=20, range=(-4, 0), feature_names=['absahv'], epochs=new_wake_ep
        ),
        "pitch" : nap.compute_tuning_curves(
            spikes, pitch, range = (-0.9, 1.5), bins=20, feature_names=['pitch'], epochs=new_wake_ep
        ),
        "roll" : nap.compute_tuning_curves(
            spikes, roll, range = (-1.5, 1.5), bins=20, feature_names=['roll'], epochs=new_wake_ep
        ),
        "pitchroll" : nap.compute_tuning_curves(
            spikes, pitch_roll, range=[(-0.9, 1.4),(-1.5, 1.5)], bins=[15, 15], feature_names=['pitch', 'roll'], epochs=new_wake_ep
        ),
        "linspeed" : nap.compute_tuning_curves(
            spikes, lin_speed, bins=20, range=(-3.5, -0.5), feature_names=['speed'], epochs=new_wake_ep
        ),
        "location": spikes.location,
        "maxch": spikes.maxch,
        "cell_type": spikes.cell_type,
        "meanwavef": tmp,
        "group": spikes.group,
        "ahv_values": logahv.restrict(new_wake_ep).values,
        "absahv_values": absahv.restrict(new_wake_ep).values,
        "pitch_values": pitch.restrict(new_wake_ep).values,
        "roll_values": roll.restrict(new_wake_ep).values,
        "linspeed_values": lin_speed.restrict(new_wake_ep).values,
    }

    ##############################################################################################
    # MUTUAL INFORMATION
    ##############################################################################################
    MI = {}
    for feat in ["hd", "ahv", "absahv", "pitch", "roll", "linspeed"]:
        SI = nap.compute_mutual_information(tuning_curves[s][feat])
        MI[feat] = SI["bits/sec"]
    MI = pd.DataFrame.from_dict(MI)
    tuning_curves[s]["MI"] = MI

    # #############################################################################################
    # # Manifold
    # #############################################################################################
    # count = spikes[spikes.location=="dg"].count(0.03, position.time_support).smooth(0.08)
    # if count.shape[1] < 2:
    #     continue
    # pca = PCA(n_components=2)
    # # pca = Isomap(n_components=2, n_neighbors=10)
    # proj = pca.fit_transform(count.values)
    # pitch_at_manifold = pitch.bin_average(0.03, position.time_support)
    # lin_speed_at_manifold = lin_speed.bin_average(0.03, position.time_support)
    # manifolds[s] = nap.TsdFrame(
    #     t=count.t,
    #     d=np.column_stack([proj, pitch_at_manifold.values, lin_speed_at_manifold.values]),
    #     columns=['pc1', 'pc2', 'pitch', 'linspeed'],
    #     time_support=count.time_support
    # )



#################################################################################################
# Plot tuning curves — one session per page
#################################################################################################
from matplotlib.backends.backend_pdf import PdfPages

colors = {"adn":"red", "ca1":"green", "dg":"#ff7f0e", "hpc":"#2ca02c", "mossy":"purple", "granule":"brown"}
ncols = 3

pdf_path = "/home/gviejo/Dropbox/UFOPhysio/figures/Tuning_curves_DG.pdf"





with PdfPages(pdf_path) as pdf:
    #############################################################################################
    # Summary figure: imshow of all tuning curves for ahv, pitch, roll (ca1 vs dg)
    #############################################################################################
    features = ["hd", "ahv", "absahv", "pitch", "roll", "linspeed"]
    coord_names = {"hd": "angle", "ahv": "ahv", "absahv": "absahv", "pitch": "pitch", "roll": "roll", "linspeed": "speed"}
    groups = ["ca1", "dg", "mossy", "granule"]
    group_fields = {"ca1": "location", "dg": "location", "mossy": "cell_type", "granule": "cell_type"}

    fig = figure(figsize=(15, 17))
    fig.suptitle("Summary tuning curves (ca1 vs dg vs mossy vs granule)", fontsize=16, y=1.01)

    gs = GridSpec(1 + len(groups), len(features), height_ratios=[1] + [2] * len(groups), wspace=0.3, hspace=0.4, figure=fig)

    for j, feat in enumerate(features):
        ax_mean = subplot(gs[0, j])
        processed = {}
        x = None

        for grp in groups:
            data = []
            for s in datasets:
                if s not in tuning_curves:
                    continue
                field = tuning_curves[s][group_fields[grp]]
                neuron_ids = field[field == grp].index.values
                if len(neuron_ids) == 0:
                    continue
                tc = tuning_curves[s][feat].sel(unit=neuron_ids)
                if x is None:
                    x = tc.coords[coord_names[feat]].values
                    if feat == "hd":
                        x = x - x[len(x) // 2]
                data.append(tc.values)

            if len(data) == 0:
                continue

            data = np.nan_to_num(np.concatenate(data, axis=0), nan=0.0)

            if feat == "hd":
                data = smooth_circular_tuning_curves(data)
                data = data / (data.max(axis=1, keepdims=True) + 1e-12)
                center = data.shape[1] // 2
                shift = center - np.argmax(data, axis=1)
                data = np.array([np.roll(row, sh) for row, sh in zip(data, shift)])
            else:
                data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
                data = gaussian_filter(data, sigma=TUNING_CURVE_SMOOTH_SIGMA)
            # order = np.argsort(np.argmax(data, axis=1))
            # data = data[order]

            processed[grp] = data

            m = data.mean(axis=0)
            sem = data.std(axis=0) / np.sqrt(data.shape[0])
            ax_mean.plot(x, m, color=colors[grp], label=grp)
            ax_mean.fill_between(x, m - sem, m + sem, color=colors[grp], alpha=0.2)

        if x is not None:
            ax_mean.set_xlim(x[0], x[-1])
        ax_mean.set_xticks([])
        title_map = {"hd": "hd (peak-centered)", "ahv": "logahv"}
        ax_mean.set_title(title_map.get(feat, feat))
        ax_mean.legend(fontsize=8, frameon=False)

        for i, grp in enumerate(groups):
            ax_img = subplot(gs[i + 1, j])

            if grp not in processed:
                ax_img.set_title("{} (n=0)".format(grp))
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                continue

            data = processed[grp]

            ax_img.imshow(data, aspect='auto', cmap='jet', extent=[x[0], x[-1], data.shape[0] - 0.5, -0.5])
            ax_img.set_xlabel(title_map.get(feat, feat))
            ax_img.set_title("{} (n={})".format(grp, data.shape[0]))
            if j == 0:
                ax_img.set_ylabel("neuron # (sorted by peak)")

    pdf.savefig(fig, bbox_inches='tight')
    close(fig)

    #############################################################################################
    # Summary figure: distribution of ahv, pitch, roll values per animal
    #############################################################################################
    animals = sorted(set(os.path.basename(s).split('-')[0] for s in datasets if s in tuning_curves))
    animal_colors = {a: cm.tab10(i % 10) for i, a in enumerate(animals)}

    fig, axes = subplots(2, 5, figsize=(25, 8))
    fig.suptitle("Distribution of logahv/absahv/pitch/roll/linspeed values per animal", fontsize=16, y=1.02)

    for ax, feat in zip(axes[0], ["ahv", "absahv", "pitch", "roll", "linspeed"]):
        for a in animals:
            values = np.concatenate([
                tuning_curves[s][feat + "_values"]
                for s in datasets
                if s in tuning_curves and os.path.basename(s).split('-')[0] == a
            ])
            ax.hist(values, bins=50, density=True, histtype='step', color=animal_colors[a], label=a)
            print(f"Animal {a}, feature {feat}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}, std={values.std():.3f}")
        ax.set_xlabel("logahv" if feat == "ahv" else feat)
        ax.set_ylabel("density")
        if feat == "linspeed":
            ax.set_xlim(-4.0, 0)

    axes[0, -1].legend(fontsize=8, frameon=False, loc='upper right')

    for ax, xfeat in zip(axes[1, :2], ["absahv", "pitch"]):
        x = []
        y = []
        for s in datasets:
            if s not in tuning_curves:
                continue
            yv = tuning_curves[s][xfeat + "_values"]
            xv = tuning_curves[s]["linspeed_values"]
            n = min(len(xv), len(yv))
            x.append(xv[:n])
            y.append(yv[:n])
        x = np.concatenate(x)
        y = np.concatenate(y)
        ax.hist2d(x, y, bins=20, cmap='jet')
        ax.set_ylabel(xfeat)
        ax.set_xlabel("linspeed")
        # if xfeat == "ahv":
        #     ax.set_ylim(-6*np.pi, 6*np.pi)
        # ax.set_xlim(-4.0, 0)

    axes[1, 2].axis('off')
    axes[1, 3].axis('off')
    axes[1, 4].axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    close(fig)

    # #############################################################################################
    # # Summary figure: neural manifold (PCA) per session, colored by pitch
    # #############################################################################################
    # sessions_with_manifold = [s for s in datasets if s in manifolds]
    # n_manifolds = len(sessions_with_manifold)
    # ncols_m = 5
    # nrows_m = n_manifolds // ncols_m + int(n_manifolds % ncols_m > 0)
    #
    # fig = figure(figsize=(4 * ncols_m, 4 * nrows_m))
    # fig.suptitle("Neural manifold (PCA) colored by pitch", fontsize=16, y=1.01)
    #
    # for i, s in enumerate(sessions_with_manifold):
    #     ax = subplot(nrows_m, ncols_m, i + 1)
    #     m = manifolds[s]
    #     sc = ax.scatter(m['pc1'].values, m['pc2'].values, c=m['pitch'].values, cmap='jet', s=2)
    #     ax.set_title(s.split('/')[-1], fontsize=8)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    #
    # pdf.savefig(fig, bbox_inches='tight')
    # close(fig)
    #
    # #############################################################################################
    # # Summary figure: neural manifold (PCA) per session, colored by linspeed
    # #############################################################################################
    # fig = figure(figsize=(4 * ncols_m, 4 * nrows_m))
    # fig.suptitle("Neural manifold (PCA) colored by linspeed", fontsize=16, y=1.01)
    #
    # for i, s in enumerate(sessions_with_manifold):
    #     ax = subplot(nrows_m, ncols_m, i + 1)
    #     m = manifolds[s]
    #     sc = ax.scatter(m['pc1'].values, m['pc2'].values, c=m['linspeed'].values, cmap='jet', s=2)
    #     ax.set_title(s.split('/')[-1], fontsize=8)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    #
    # pdf.savefig(fig, bbox_inches='tight')
    # close(fig)

    #############################################################################################
    # Summary figure: PCA of neurons on concatenated ahv/pitch/linspeed tuning curves
    #############################################################################################
    def zscore_tc(v):
        return (v - v.mean()) / (v.std() + 1e-12)

    tc_features = []
    tc_locations = []

    for s in datasets:
        if s not in tuning_curves:
            continue
        for nid in tuning_curves[s]['location'].index.values:
            loc = tuning_curves[s]['location'][nid]
            if loc not in ("ca1", "dg"):
                continue
            ahv_tc = np.nan_to_num(tuning_curves[s]['ahv'].sel(unit=nid).values, nan=0.0)
            absahv_tc = np.nan_to_num(tuning_curves[s]['absahv'].sel(unit=nid).values, nan=0.0)
            pitch_tc = np.nan_to_num(tuning_curves[s]['pitch'].sel(unit=nid).values, nan=0.0)
            linspeed_tc = np.nan_to_num(tuning_curves[s]['linspeed'].sel(unit=nid).values, nan=0.0)

            # feat_vec = np.concatenate([zscore_tc(ahv_tc), zscore_tc(linspeed_tc)])
            # feat_vec = np.concatenate([absahv_tc, linspeed_tc, pitch_tc])
            feat_vec = np.concatenate([zscore_tc(absahv_tc), zscore_tc(ahv_tc)])
            tc_features.append(feat_vec)
            tc_locations.append(loc)

    tc_features = np.array(tc_features)
    tc_locations = np.array(tc_locations)

    # pca_tc = PCA(n_components=2)
    pca_tc = UMAP(n_components=2, n_neighbors=10, min_dist=0.01, random_state=42)
    proj_tc = pca_tc.fit_transform(tc_features)

    fig = figure(figsize=(20, 15))
    fig.suptitle("PCA of neurons (logahv + absahv tuning curves) and mutual information", fontsize=16, y=1.02)
    gs = GridSpec(3, len(features), hspace=0.4, wspace=0.4, figure=fig)

    ax0 = subplot(gs[0, :2])
    for grp in ["ca1", "dg"]:
        mask = tc_locations == grp
        ax0.scatter(proj_tc[mask, 0], proj_tc[mask, 1], color=colors[grp], label=grp, s=10, alpha=0.7)
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.set_title("PCA of neurons")
    ax0.legend(fontsize=8, frameon=False)

    #############################################################################################
    # Mutual information: histogram per feature
    #############################################################################################
    mi_rows = []
    for s in datasets:
        if s not in tuning_curves:
            continue
        mi_df = tuning_curves[s]["MI"]
        loc = tuning_curves[s]["location"]
        cell_type = tuning_curves[s]["cell_type"]
        for nid in mi_df.index:
            if nid not in loc.index or loc[nid] not in ("ca1", "dg"):
                continue
            row = mi_df.loc[nid].to_dict()
            row["location"] = loc[nid]
            row["cell_type"] = cell_type[nid] if nid in cell_type.index else "other"
            mi_rows.append(row)
    mi_pooled = pd.DataFrame(mi_rows)

    for j, feat in enumerate(features):
        ax = subplot(gs[1, j])
        for grp in ["ca1", "dg"]:
            sub = mi_pooled[mi_pooled["location"] == grp]
            ax.hist(sub[feat].dropna(), bins=30, density=True, histtype='step', color=colors[grp], label=grp)
        feat_label = "logahv" if feat == "ahv" else feat
        ax.set_xlabel("MI {} (bits/sec)".format(feat_label))
        ax.set_ylabel("density")
        ax.set_title(feat_label)
        if j == 0:
            ax.legend(fontsize=7, frameon=False)

    for j, feat in enumerate(features):
        ax = subplot(gs[2, j])
        for grp in ["granule", "mossy"]:
            sub = mi_pooled[mi_pooled["cell_type"] == grp]
            ax.hist(sub[feat].dropna(), bins=30, density=True, histtype='step', color=colors[grp], label=grp)
        feat_label = "logahv" if feat == "ahv" else feat
        ax.set_xlabel("MI {} (bits/sec)".format(feat_label))
        ax.set_ylabel("density")
        ax.set_title(feat_label)
        if j == 0:
            ax.legend(fontsize=7, frameon=False)

    pdf.savefig(fig, bbox_inches='tight')
    close(fig)

    ##############################################################################################
    # Plot tuning curves for each session
    ##############################################################################################

    for s in datasets:
        print("Plotting session {}".format(s))


        order = ["ca1", "dg"]
        neuron_ids = []
        for group in order:
            neuron_ids += tuning_curves[s]['location'][tuning_curves[s]['location'] == group].index.values.tolist()

        # # Sort by maxch & group
        # # neuron_ids = tuning_curves[s]['maxch'].sort_values().index.values
        # neuron_ids = []
        # for group in np.unique(tuning_curves[s]['group']):
        #     neuron_ids += tuning_curves[s]['maxch'][tuning_curves[s]['group'] == group].sort_values().index.values.tolist()
        # neuron_ids = np.array(neuron_ids)

        n = len(neuron_ids)
        if n == 0:
            continue

        nrows = n // ncols + int(n % ncols > 0)
        fig = figure(figsize=(40, nrows * 5))
        fig.suptitle(s.split('/')[-1], fontsize=16, y=1.01)

        gs = GridSpec(nrows, ncols, wspace=0.3, hspace=0.5, figure=fig)

        for j, nid in enumerate(neuron_ids):
            gs3 = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[j // ncols, j % ncols], wspace=0.3, hspace=0.5)

            color = colors[tuning_curves[s]['location'][nid]]


            ax0 = subplot(gs3[0, 0])
            meanwavef = tuning_curves[s]['meanwavef'][nid]
            cmap = 'viridis' if tuning_curves[s]['location'][nid] == 'ca1' else 'plasma'
            imshow(meanwavef.T, aspect='auto', cmap=cmap)
            if tuning_curves[s]['location'][nid] in ['ca1', 'dg']:
                axhline(channels_separation[s.split('/')[-1]], color='w', linestyle='--')
            xticks([])
            yticks([0, meanwavef.shape[1]//2, meanwavef.shape[1]-1], [0, meanwavef.shape[1]//2, meanwavef.shape[1]])
            title("group: {}, location: {}".format(tuning_curves[s]['group'][nid], tuning_curves[s]['location'][nid]))

            ax1 = subplot(gs3[0, 1], projection='polar')
            tc = tuning_curves[s]['hd'].sel(unit=nid)
            angle = tc.coords['angle'].values
            smoothed = smooth_circular_tuning_curves(np.nan_to_num(tc.values, nan=0.0)[None, :])[0]
            ax1.step(angle, smoothed, where='mid', color=color)
            xticks([])
            # yticks([])

            ax1b = subplot(gs3[0, 2])
            tc = tuning_curves[s]['linspeed'].sel(unit=nid)
            x = tc.coords['speed'].values
            smoothed = gaussian_filter1d(np.nan_to_num(tc.values, nan=0.0), sigma=TUNING_CURVE_SMOOTH_SIGMA)
            ax1b.step(x, smoothed, where='mid', color=color)
            title("")
            xlabel("linspeed")

            ax2 = subplot(gs3[1, 0])
            tc = tuning_curves[s]['ahv'].sel(unit=nid)
            x = tc.coords['ahv'].values
            y = np.nan_to_num(tc.values, nan=0.0)
            ax2.step(x, y, where='mid', color=color)
            # xticks([])
            title("")
            xlabel("logahv")

            ax2b = subplot(gs3[0, 3])
            tc = tuning_curves[s]['absahv'].sel(unit=nid)
            x = tc.coords['absahv'].values
            smoothed = gaussian_filter1d(np.nan_to_num(tc.values, nan=0.0), sigma=TUNING_CURVE_SMOOTH_SIGMA)
            ax2b.step(x, smoothed, where='mid', color=color)
            title("")
            xlabel("absahv")

            ax3 = subplot(gs3[1, 1])
            tc = tuning_curves[s]['pitch'].sel(unit=nid)
            x = tc.coords['pitch'].values
            smoothed = gaussian_filter1d(np.nan_to_num(tc.values, nan=0.0), sigma=TUNING_CURVE_SMOOTH_SIGMA)
            ax3.step(x, smoothed, where='mid', color=color)
            # xticks([])
            title("")
            xlabel("pitch")

            ax4 = subplot(gs3[1, 2])
            tc = tuning_curves[s]['roll'].sel(unit=nid)
            x = tc.coords['roll'].values
            smoothed = gaussian_filter1d(np.nan_to_num(tc.values, nan=0.0), sigma=TUNING_CURVE_SMOOTH_SIGMA)
            ax4.step(x, smoothed, where='mid', color=color)
            # xticks([])
            title("")
            xlabel("roll")



        subplots_adjust(top=0.95, bottom=0.02)
        pdf.savefig(fig, bbox_inches='tight')
        close(fig)

    # # Plot maxch distribution
    # n = len(ch_pos)
    # ncols = 4
    # nrows = n // ncols + int(n % ncols