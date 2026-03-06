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
from scipy.signal import correlate, correlation_lags

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

err_ahvs = {
    "ufo": [],
    "ds": []
    }


def cross_corr(x, y):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    corr = correlate(x, y, mode='same', method='fft') / len(x)
    lags = correlation_lags(len(x), len(y), mode='same')

    return lags, corr

# for s in datasets:
# for s in ['ADN-HPC/B5100/B5101/B5101-250502']:
for s in ["ADN-HPC/B5100/B5102/B5102-250919"]:
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

    # Waveform classification
    spikes.location = [v.lower() for v in spikes.location.values]
    meanwavef, maxch = load_mean_waveforms(path)
    maxchs.append(maxch)
    spikes.maxch = np.hstack([chs for chs in maxch.values()])
    location = spikes.location.copy()
    location[(spikes.maxch<30) & (spikes.location == "hpc")] = "ca1"
    location[(spikes.maxch>=30) & (spikes.location == "hpc")] = "dg"
    spikes.location = location

    spikes = spikes[(spikes.location == 'adn') | (spikes.group == 0)]
    # spikes = spikes[spikes.group == 0]

    # spikes = spikes[spikes.rate > 1.0]

    ahv = np.unwrap(position['ry']).bin_average(0.02, position.time_support).derivative()

    # new_wake_ep = wake_ep.intersect(np.abs(ahv).smooth(0.1).threshold(0.3).time_support)

    ahv_tuning = nap.compute_tuning_curves(
        spikes, ahv, bins=20, range=(-2*np.pi, 2*np.pi), feature_names=['ahv'], epochs=wake_ep
    )

    # spikes = spikes[spikes.ahv_mi > 0.03]
    # ahv_tuning = ahv_tuning.sel(unit=spikes.keys())

    # decoded, P = nap.decode_bayes(
    #     ahv_tuning,
    #     data=spikes,
    #     epochs=wake_ep,
    #     bin_size=0.02,
    #     sliding_window_size=5,
    #     uniform_prior=True
    # )

    X = spikes.count(0.02, position.time_support).smooth(0.04)

    X_dg = X[:, spikes.group==0]
    X_adn = X[:, spikes.location=="adn"]

    # Learning the ahv model from DG activity
    model_ahv = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=6,
        objective='reg:squarederror'  # default for regression
    )
    model_ahv.fit(X_dg.d, ahv.values)
    Yp = model_ahv.predict(X_dg.d)
    Yp = nap.Tsd(t = X_dg.index.values, d = Yp, time_support = X_dg.time_support)

    model_hd = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.01))

    Y = np.column_stack([np.cos(position['ry']), np.sin(position['ry'])])
    Y = Y.bin_average(0.02, position.time_support)

    model_hd.fit(X_adn.d, Y)
    Yp_hd = model_hd.predict(X_adn.d)
    Yp_hd = nap.TsdFrame(t=X_adn.index.values, d=Yp_hd, time_support=X_adn.time_support)
    hd_pred = nap.Tsd(t=Yp_hd.index.values, d=np.arctan2(Yp_hd.d[:, 1], Yp_hd.d[:, 0]), time_support=Yp_hd.time_support)

    model_ahv2 = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=6,
        objective='reg:squarederror'  # default for regression
    )
    model_ahv2.fit(X[:, spikes.location=="adn"].d, ahv.values)
    ahv2_adn = model_ahv2.predict(X[:, spikes.location=="adn"].d)
    ahv2_adn = nap.Tsd(t=X[:, spikes.location=="adn"].index.values, d=ahv2_adn, time_support=X.time_support)


    # DECODING DURING SLEEP
    Xs = spikes.count(0.01, sws_ep).smooth(0.04)

    Yp_hd = model_hd.predict(Xs[:, spikes.location=="adn"].d)
    hd_pred = nap.Tsd(t=Xs.t, d=np.arctan2(Yp_hd[:, 1], Yp_hd[:, 0]) % (2*np.pi), time_support=Xs.time_support)
    ahv_adn = np.unwrap(hd_pred).derivative()

    # ahv_adn = model_ahv2.predict(Xs[:, spikes.location=="adn"].d)
    # ahv_adn = nap.Tsd(t=Xs.t, d=ahv_adn, time_support=Xs.time_support)


    ahv_dg = model_ahv.predict(Xs[:, spikes.group==0].d)
    ahv_dg = nap.Tsd(t=Xs.t, d=ahv_dg, time_support=Xs.time_support)

    ahv_sleep = np.column_stack([ahv_adn, ahv_dg])
    ahv_sleep.location = ["adn", "dg"]


    # err = np.power(ahv_adn.values - ahv_dg.values, 2)
    # err = np.abs(ahv_adn.values - ahv_dg.values)
    # err = nap.Tsd(t=ahv_adn.index.values, d=err, time_support=ahv_adn.time_support)

    _, err = cross_corr(ahv_adn.restrict(sws_ep).values, ahv_dg.restrict(sws_ep).values)
    err = nap.Tsd(t=ahv_adn.t, d=err, time_support=ahv_adn.time_support)

    cc_ufo = nap.compute_perievent_continuous(err, ufo_ts.restrict(sws_ep), (-0.5, 0.5), ep=sws_ep)
    cc_ufo = cc_ufo.as_dataframe()
    cc_ds = nap.compute_perievent_continuous(err, ds_ts.restrict(sws_ep), (-0.5, 0.5), ep=sws_ep)
    cc_ds = cc_ds.as_dataframe()

    err_ahvs["ufo"].append(cc_ufo.mean(1))
    err_ahvs["ds"].append(cc_ds.mean(1))

    # cc_ufo = cc_ufo - cc_ufo.mean(0)
    # cc_ufo = cc_ufo / cc_ufo.std(0)
    # cc_ds = cc_ds - cc_ds.mean(0)
    # cc_ds = cc_ds / cc_ds.std(0)

for k in err_ahvs.keys():
    err_ahvs[k] = pd.concat(err_ahvs[k], axis=1)


# 1023

figure()
for i, k in enumerate(err_ahvs.keys()):
    subplot(1, 2, i+1)
    tmp = err_ahvs[k]
    # tmp = tmp - tmp.mean(0)
    # tmp = tmp / tmp.std(0)
    plot(tmp, alpha=0.3)
    plot(tmp.mean(1), linewidth=2, color='k')
    title("Error between ADN and DG AHV around " + k)
    xlabel("Time from {} (s)".format(k.upper()))
    grid()
legend()
show()

figure()
for i in range(len(ahv_tuning)):
    subplot(7, 5, i+1)
    plot(ahv_tuning.ahv.values, ahv_tuning.values[i])

figure()
plot(position['ry'])
plot(hd_pred%(2*np.pi))

figure()
plot(ahv)
plot(Yp)
plot(ahv2_adn)
show()