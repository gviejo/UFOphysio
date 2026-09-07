# Overview of `main_pyna_*` scripts

Analysis entry-point scripts (pynapple-based) for the UFO physiology project.
Most scripts iterate over a `datasets_*.list` file, load a session with `nwbmatic`,
compute some metric, and save a figure (PDF/PNG) to `~/Dropbox/UFOPhysio/figures/`
and/or a `.pickle` file for later pooling/plotting. Grouped below by topic.
(One non-`main_pyna_`-named script, `main_sound_click.py`, is included below since it
belongs to the same analysis pipeline.)

## UFO detection & basic preprocessing

| Script | Description |
|---|---|
| `main_pyna_LMN_detect_UFO.py` | Detects UFO events from LMN LFP channels across sessions (core UFO detector). |
| `main_pyna_PAIRED_LMN_ADN_detect_UFO.py` | Compares UFO detection run independently on LMN vs ADN channels for the same sessions, to check the two detectors agree; saves `Pairing_UFO_ADN-LMN_0/1.pdf`. |
| `main_pyna_LMN_detect_SWR.py` | Detects sharp-wave ripples (SWR) in LMN/hippocampal LFP. |
| `main_pyna_LMN_detect_DS.py` | Detects dentate spikes (DS) via `ds_detection.py` and produces a per-session PDF summary (`DS_detection_summary.pdf`). |
| `main_pyna_UFO_viewer.py` | Interactive/example viewer that plots UFO-triggered decoding examples and stitches per-session PDFs with `pdftk`. |
| `main_pyna_spiking_circus_dat_file.py` | Prepares/exports a `.dat` file for a session to be spike-sorted with SpyKING CIRCUS. |

## UFO temporal/spectral properties

| Script | Description |
|---|---|
| `main_pyna_UFO_FREQUENCY_IUI.py` | Computes UFO occurrence frequency and inter-UFO-interval (IUI) distributions; saves `Frquency_UFO.png`, `IUI_UFO.png`. |
| `main_pyna_UFO_frequency_sessions.py` | Compares UFO frequency across sessions/animals (`UFO_across_sessions.pdf`). |
| `main_pyna_UFO_autocorrs.py` | UFO event autocorrelograms (`Auto_corrs_UFO.png`). |
| `main_pyna_UFO_SPECT.py` | Spectrogram of LFP around UFO events (`UFO_SPECTROGRAM.png`). |
| `main_pyna_UFO_Spectral_Entropy.py` | Computes spectral entropy of LFP around UFOs for LMN/ADN/PSB. |
| `main_pyna_LMN_SPECT.py` | General LMN LFP spectrogram analysis. |
| `main_pyna_LMN_STFT.py` | Short-time Fourier transform (multitaper) analysis of LMN LFP. |
| `main_pyna_PSD_LMN.py` | Power spectral density (multitaper) of LMN LFP. |
| `main_pyna_UFO_mean_HPC_LFP.py` | Mean hippocampal LFP waveform triggered on UFO events (`UFO_mean_HPC_LFP.pdf`). |
| `main_pyna_UFO_offset_tc.py` | Tuning-curve-like analysis of UFO channel/detection offsets (KernelPCA on offset signatures). |
| `main_pyna_test_emd.py` | Scratch/test script exploring Empirical Mode Decomposition on LFP signals. |

## UFO cross-correlations with other events/units

| Script | Description |
|---|---|
| `main_pyna_UFO_ALL_cross_corr.py` | Aggregates spike/unit cross-correlograms with UFOs across ADN, LMN, and PSB regions (`ALL_CC_UFO_Long/Short.png`). |
| `main_pyna_UFO_DS_cross_corr.py` | Cross-correlation between UFOs and dentate spikes, plus UFO-DG spike cross-corr. |
| `main_pyna_UFO_SWR_cross_corr.py` | Cross-correlation between UFOs and sharp-wave ripples. |
| `main_pyna_UFO_MMN_cross_corr.py` | Cross-correlation between UFOs, LMN and mammillary nuclei (MMN) unit activity. |
| `main_pyna_UFO_PSB_cross_corr.py` | UFO-triggered cross-correlograms and tuning curves for post-subiculum (PSB) units in an example session. |
| `main_pyna_UFO_UP_DOWN_CC.py` | Cross-correlation of UFOs with cortical UP/DOWN state transitions (ADN & PSB sessions). |
| `main_pyna_ADN_DG_cross_corr.py` | Unit cross-correlograms and tuning curves for ADN/DG/CA1 sessions (`Cross_corr_ADN_HPC/DG.pdf`). |
| `main_pyna_B51_CC_UFO.py` | Mean unit-UFO cross-correlograms (wake & SWS) for the "B51" dataset, split by peak-positive/negative modulation. |
| `main_pyna_search_UFO_SWR_example.py` | Finds/plots example sessions/epochs showing both UFOs and SWRs together. |
| `main_pyna_search_SPECT_examples.py` | Finds/saves example LMN-ADN spectrogram snippets as PNGs. |

## Sound stimulation (`click/`)

| Script | Description |
|---|---|
| `click/main_sound_click.py` | Computes cross-correlograms and PETHs of ADN/LMN/DS units aligned to auditory click-sound stimulus onsets (wake & SWS), per anatomical structure; saves `Cross-corr_sound.pdf` and pools results into `cc_sound.pickle`. |

## Correlation with behavior (head direction, angular/linear velocity, theta)

| Script | Description |
|---|---|
| `main_pyna_corr_UFO_hd.py` | Correlates UFO occurrence/rate with head-direction (HD) tuning offset (`figUFO_AHV_tc_offset.pdf`). |
| `main_pyna_corr_UFO_ahv_speed.py` | Correlates UFO occurrence with angular head velocity (AHV) and linear speed tuning curves. |
| `main_pyna_corr_UFO_speed.py` | Correlates UFO rate with linear running speed; saves `CORR_UFO_SPEED.pickle`. |
| `main_pyna_corr_UFO_THETA.py` | Relates UFO occurrence to theta oscillation phase/power (`Corr_UFO_Theta.png`). |
| `main_pyna_ADN_DG_AHV_CC.py` | Cross-correlation between AHV and UFOs/units for combined ADN-DG sessions (dimensionality reduction + XGBoost regressors). |
| `main_pyna_ADN_DG_AHV_decoding.py` | Decodes |AHV| peri-UFO during SWS from ADN/DG population activity; per-session and pooled PDF figures. |
| `main_pyna_PAIRED_UFO_AHV.py` | AHV-related analysis for paired LMN-ADN UFO detections (`Pairing_UFO_ADN-LMN_2/3/4.pdf`). |
| `main_pyna_NSS_AHV_CORR.py` | Paired-UFO AHV correlation analysis (variant), saves `Pairing_UFO_ADN-LMN_5/6.pdf`. |
| `main_pyna_LMN_ADN_correlation.py` | Pearson-correlation of tuning curves (e.g. HD) between LMN and ADN units across sessions. |
| `main_pyna_ADN_correlation.py` | Pearson-correlation of tuning curves for ADN units across sessions/epochs. |
| `main_pyna_MB_control.py` | Control analysis using mammillary body (MB) channels/power as a comparison for UFO-related signals. |

## Reactivation analyses

| Script | Description |
|---|---|
| `main_pyna_ADN_DG_reactivation.py` | UFO-triggered explained-variance/reactivation analysis for ADN/DG/CA1/HPC populations; pools and saves `UFO_reactivation_ADN_DG.pdf`. |
| `main_pyna_ADN_DG_reactivation_aversive.py` | Same reactivation analysis restricted to aversive-task sessions. |
| `main_pyna_LMN_ADN_reactivation.py` | UFO-triggered reactivation for LMN/ADN, including UP-state-onset and jitter-control comparisons (`UFO_reactivation_LMN_ADN.pdf`). |
| `main_pyna_LMN_PSB_reactivation.py` | Same reactivation analysis for LMN/PSB sessions. |
| `main_pyna_LMN_PSB_reactivation_aversive.py` | LMN/PSB reactivation analysis restricted to aversive-task sessions. |
| `main_pyna_UFO_reactivation_summary.py` | Pools the pickled reactivation results from the scripts above (ADN/DG/CA1/HPC/LMN/PSB) into one summary figure. |

## Decoding (`decoding/`)

| Script | Description |
|---|---|
| `decoding/main_pyna_ADN_UFO_decoding.py` | Bayesian/population decoding around UFO events using ADN units. |
| `decoding/main_pyna_UFO_wake_decoding.py` | Decodes behavioral variables during wake using a GradientBoosting classifier trained on UFO-adjacent activity. |
| `decoding/main_pyna_UFO_DS_decoding.py` | Decoding analysis relating UFOs to dentate spikes (DS). |
| `decoding/main_pyna_UFO_ADN-LMN_decoding.py` | Decodes angular speed / state around UFOs from combined ADN-LMN populations (`UFO_ADN-LMN_decoding_angspeed.pdf`). |
| `decoding/main_pyna_UFO_ADN-LMN_decoding_examples.py` | Generates example-session figures for the ADN-LMN UFO decoding above (uses `pynaviz` for interactive scoping). |
| `main_pyna_ADN_DG_AHV_decoding.py` | (see behavior table above) — AHV decoding for ADN-DG. |

## Clustering / spatial info (`clustering/`)

| Script | Description |
|---|---|
| `clustering/main_pyna_UFO_clustering.py` | Clusters UFO waveforms/features (KernelPCA, HDBSCAN/KMeans) to look for UFO subtypes. |
| `clustering/main_pyna_UFO_spatial_info.py` | Computes spatial/positional information content of units relative to UFOs (`UFO_spatial_info.pdf`). |

## Dentate gyrus (DG) cell classification & tuning

| Script | Description |
|---|---|
| `main_pyna_DG_classify_cells.py` | Classifies DG units into subtypes (interneuron, granule/mossy cell, etc.) from waveform + firing features via PCA; writes per-session classification files and a QC PDF (`Cell_classification_DG.pdf`). |
| `main_pyna_DG_tunings.py` | Computes and plots tuning curves (HD, AHV, linear speed, pitch/roll) for DG/CA1/mossy/granule cells (`Tuning_curves_DG.pdf`). |

## B51-specific analyses

| Script | Description |
|---|---|
| `main_pyna_B51_tuning_curves.py` | Tuning curves (HD/AHV) for units in the "B51" dataset, one page per session (`Tuning_curves_B51_ADDG.pdf`). |
| `main_pyna_B51_CC_UFO.py` | (see cross-correlation table above) — B51 unit-UFO cross-correlograms. |

## Paired UFO examples

| Script | Description |
|---|---|
| `main_pyna_PAIRED_UFO_examples.py` | Plots example traces of paired LMN/ADN UFO detections (`Pairing_UFO_ADN-LMN_7/8_examples.pdf`). |

Note: `main_pyna_PAIRED_LMN_ADN_detect_UFO.py`, `main_pyna_PAIRED_UFO_AHV.py`, and `main_pyna_NSS_AHV_CORR.py` are also part of this "paired UFO" family (cross-referenced above under detection and behavior).