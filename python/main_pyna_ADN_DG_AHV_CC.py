import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
from matplotlib.gridspec import GridSpecFromSubplotSpec

from functions.functions import load_mean_waveforms
from ufo_detection import *
from matplotlib.pyplot import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

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

datasets = np.genfromtxt(os.path.join(data_directory,'datasets_ADN_DG.list'), delimiter = '\n', dtype = str, comments = '#')

maxchs = []

corr_ahv_cc = []
ahvs = []
ccs = []

ahv_ds_corr = {}
ahv_ufo_corr = {}
ahv_ds2_corr = {}

for s in datasets:
# for s in ['ADN-HPC/B5100/B5101/B5101-250502']:
    print(s)
    ###############################################################################################
    # LOADING DATA
    ###############################################################################################
    path = os.path.join(data_directory, s)
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    position = data.position
    wake_ep = data.epochs['wake']
    sws_ep = data.read_neuroscope_intervals('sws')
    #rem_ep = data.read_neuroscope_intervals('rem')
    ufo_ep, ufo_ts = loadUFOs(path)
    ds_ep, ds_ts = loadDentateSpikes(path)
    ds_ts = nap.Ts(ds_ep.start)

    # Waveform classification
    spikes.location = [v.lower() for v in spikes.location.values]
    meanwavef, maxch = load_mean_waveforms(path)
    maxchs.append(maxch)
    spikes.maxch = np.hstack([chs for chs in maxch.values()])
    location = spikes.location.copy()
    location[(spikes.maxch<30) & (spikes.location == "hpc")] = "ca1"
    location[(spikes.maxch>=30) & (spikes.location == "hpc")] = "dg"
    spikes.location = location

    # spikes = spikes[(spikes.location == 'adn') | (spikes.group == 0)]
    # spikes = spikes[spikes.group == 0]

    # spikes = spikes[spikes.rate > 1.0]

    ahv = np.unwrap(position['ry']).bin_average(0.02, position.time_support).derivative()

    new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)


    ###############################################################################################
    # CORRELATION AHV TUNING VS PEAK CC UFO/DG
    ###############################################################################################
    dg_spikes = spikes[spikes.group == 0]

    ahv_tuning = nap.compute_tuning_curves(
        dg_spikes, ahv, bins=20, range=(-np.pi, np.pi), feature_names=['ahv'], epochs=new_wake_ep,
        return_pandas=True
    )
    # ahv_tuning = nap.compute_tuning_curves(
    #     spikes, ahv, bins=15, range=(0, 2*np.pi), feature_names=['ahv'], epochs=wake_ep,
    #     return_pandas=True
    # )

    a, b = np.polyfit(ahv_tuning.index.values, ahv_tuning.values, deg=1)

    # for i in range(len(ahv_tuning)):
    #     subplot(5, 5, i+1)
    #     plot(ahv_tuning.ahv.values, ahv_tuning.values[i], 'o')
    #     plot(ahv_tuning.ahv.values, a[i]*ahv_tuning.ahv.values + b[i], 'r-')

    cc_dg = nap.compute_eventcorrelogram(
        dg_spikes, ufo_ts, 0.002, 0.1, sws_ep, norm=True
    )

    # Peak of the cross-correlogram in the first 30 ms
    tmp = cc_dg.loc[0:0.03]
    # pk = cc_dg.loc[0:0.03].max()
    pk = np.sum(tmp - tmp.min(0), 0)

    tmp = pd.DataFrame({
        "a": a,
        "b": b,
        "cc": pk.values
    })
    tmp.index = [s.split("/")[-1] + "_" + str(n) for n in dg_spikes.keys()]

    ahv_tuning.columns = tmp.index
    cc_dg.columns = tmp.index

    corr_ahv_cc.append(tmp)
    ccs.append(cc_dg)
    ahvs.append(ahv_tuning)

    #####################################################################################################
    # PERIEVENT AHV SLEEP / DS
    #######################################################################################################
    spikes_adn = spikes[spikes.location == "adn"]
    tuning_curves = nap.compute_tuning_curves(
        spikes_adn, position['ry'], 60, range=(0, 2 * np.pi), epochs=position.time_support
    )
    decoded, P = nap.decode_bayes(
        tuning_curves,
        data=spikes_adn,
        epochs=sws_ep,
        bin_size=0.01,
        sliding_window_size=5,
        uniform_prior=True
    )
    ahv_pred = np.unwrap(decoded).derivative()

    # X = spikes_adn.count(0.05, position.time_support).smooth(0.3)
    # model_hd = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.01))
    # Y = np.column_stack([np.cos(position['ry']), np.sin(position['ry'])])
    # Y = Y.bin_average(0.05, position.time_support)
    # model_hd.fit(X.d, Y.d)
    # Xs = spikes_adn.count(0.01, sws_ep).smooth(0.03)
    # pred = model_hd.predict(Xs.d)
    # hd_pred = nap.Tsd(t=Xs.index.values, d=np.arctan2(pred[:, 1], pred[:, 0]), time_support=Xs.time_support)
    # ahv_pred = hd_pred.derivative()

    # Separating DS+ and DS- events
    ufo_tsd = nap.Tsd(t=ufo_ts.t, d=ufo_ts.t, time_support=ufo_ts.time_support)
    ds_tsd = ds_ts.value_from(ufo_tsd, mode='before')
    dsufo_tsd = ds_tsd[np.abs(ds_tsd - ds_tsd.t)<0.1]
    dsnoufo_tsd = ds_tsd[np.abs(ds_tsd - ds_tsd.t)>=0.1]

    dpec = nap.compute_perievent_continuous(ahv_pred, dsufo_tsd.restrict(sws_ep), (-0.3, 0.3), ep=sws_ep)
    dpec = dpec.as_dataframe()

    ahv_ds_corr[s] = dpec.mean(1)

    dpec2 = nap.compute_perievent_continuous(ahv_pred, dsnoufo_tsd.restrict(sws_ep), (-0.3, 0.3), ep=sws_ep)
    dpec2 = dpec2.as_dataframe()

    ahv_ds2_corr[s] = dpec2.mean(1)



corr_ahv_cc = pd.concat(corr_ahv_cc, axis=0)
ccs = pd.concat(ccs, axis=1)
ahvs = pd.concat(ahvs, axis=1)
ahv_ds_corr = pd.DataFrame(ahv_ds_corr)
ahv_ds2_corr = pd.DataFrame(ahv_ds2_corr)

#pc = PCA(n_components=2).fit_transform(ahvs.values.T)
pc = TSNE(n_components=2).fit_transform(ahvs.values.T)

figure(figsize=(15, 5))
subplot(1, 3, 1)
plot(corr_ahv_cc.a, corr_ahv_cc.cc, 'o', markersize=2)
xlabel("AHV tuning slope")
ylabel("Peak CC UFO/DG")
subplot(1, 3, 2)
scatter(pc[:, 0], pc[:, 1], s=3, c=np.log(corr_ahv_cc.cc), cmap='viridis', vmin=np.log(corr_ahv_cc.cc).min(), vmax=2.5)
xlabel("PC1 AHV tuning")
ylabel("PC2 AHV tuning")
tight_layout()
subplot(1, 3, 3)
# ahv_ds_corr = ahv_ds_corr - ahv_ds_corr.mean(0)
# ahv_ds_corr = ahv_ds_corr / ahv_ds_corr.std(0)
plot(ahv_ds_corr)
plot(ahv_ds_corr.mean(1), 'k-')
xlabel("Time from DS (s)")

savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_ADN-DG_correlation_ahv_cc_summary.pdf"), dpi=100)



corr_ahv_cc = corr_ahv_cc.sort_values("cc")

figure(figsize=(40, 100))
n_cols = 6
gs = GridSpec(len(corr_ahv_cc)//n_cols + 1, n_cols, wspace=0.3, hspace=0.3)
for i, n in enumerate(corr_ahv_cc.index):
    gs2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i//n_cols, i%n_cols], wspace=0.3)
    subplot(gs2[0, 0])
    plot(ccs[n], 'k-')
    axvline(0, color='r', lw=1)
    subplot(gs2[0, 1])
    plot(ahvs[n])
    plot(ahvs[n].index, corr_ahv_cc.loc[n, "a"]*ahvs[n].index + corr_ahv_cc.loc[n, "b"], 'r-')
    title(n)

tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_ADN-DG_correlation_ahv_cc.pdf"), dpi=100)

figure(figsize=(10, 20))
n_cols = 3
gs = GridSpec(ahv_ds_corr.shape[1]//n_cols + 1, n_cols, wspace=0.3, hspace=0.3)
for i, s in enumerate(ahv_ds_corr.columns):
    subplot(gs[i//n_cols, i%n_cols])
    plot(ahv_ds_corr[s], label="DS+UFO")
    plot(ahv_ds2_corr[s], label="DS-UFO")

    xlabel("Time from DS (s)")
    title(s.split("/")[-1])
    legend()
tight_layout()
savefig(os.path.expanduser("~/Dropbox/UFOPhysio/figures/UFO_ADN-DG_correlation_ahv_perievent.pdf"), dpi=100)

