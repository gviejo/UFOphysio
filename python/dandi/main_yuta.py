from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
import pandas as pd
from matplotlib.pyplot import *

dandiset_id = "000003"

# Cache streamed bytes locally so re-runs / repeated access don't re-download
fs = fsspec.filesystem("http")
fs = CachingFileSystem(fs=fs, cache_storage="/tmp/nwb-cache")

labels = ['granule cell', 'mossy cell', 'narrow waveform cell', 'unknown']

# One list of tuning curves (DataFrame, columns=units) per cell-type label,
# collected across every session of every animal in the dandiset
ahv_tc = {lb: [] for lb in labels}

# Number of cells per category, per session
cell_counts = []

with DandiAPIClient() as client:
    dandiset = client.get_dandiset(dandiset_id, "draft")
    assets = [a for a in dandiset.get_assets() if a.path.endswith(".nwb")]

    for asset in assets:
        print("Processing", asset.path)
        io = None
        try:
            url = asset.get_content_url(follow_redirects=1, strip_query=True)
            file = h5py.File(fs.open(url, "rb"))
            io = NWBHDF5IO(file=file, load_namespaces=True)
            nwb = nap.NWBFile(io.read())

            spikes = nwb['units']
            groups = {lb: spikes[spikes.cell_type == lb] for lb in labels}

            counts = {lb: len(groups[lb]) for lb in labels}
            counts["session"] = asset.path
            cell_counts.append(counts)
            print("  cell counts:", counts)

            position = nwb['position_sensor0']
            epochs = nwb['states']
            wake_ep = epochs[epochs.label == "awake"]

            # Compute angle from position
            dpos = position.bin_average(0.1)
            heading = np.arctan2(np.diff(dpos.loc['y'].values), np.diff(dpos.loc['x'].values)) + np.pi
            heading = nap.Tsd(t=dpos.index.values[1:], d=heading, time_support=position.time_support)
            heading = heading.dropna()
            heading = np.unwrap(heading)
            heading = heading.smooth(0.2)
            heading = heading.restrict(wake_ep)
            # Compute ahv
            ahv = heading.derivative()

            # Compute tuning curves for each group
            for lb in labels:
                group = groups[lb]
                if len(group) == 0:
                    continue
                tc = nap.compute_tuning_curves(
                    group, ahv, bins=120, range=(-6 * np.pi, 6 * np.pi), return_pandas=True
                )
                ahv_tc[lb].append(tc)
        except Exception as e:
            print("  skipped:", e)
        finally:
            if io is not None:
                io.close()

# Cell counts per category, per session
counts_df = pd.DataFrame(cell_counts).set_index("session")
print(counts_df)

# Concatenate tuning curves from all sessions/animals, cell by cell
for lb in labels:
    if len(ahv_tc[lb]):
        ahv_tc[lb] = pd.concat(ahv_tc[lb], axis=1)

figure()
for i, lb in enumerate(labels):
    subplot(2, 2, i + 1)
    tc = ahv_tc[lb]
    if len(tc):
        plot(tc.mean(1))
    title(lb)
    xlabel("Angular velocity (rad/s)")
    ylabel("Firing rate (Hz)")
tight_layout()
show()