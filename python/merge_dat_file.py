# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
"""
Concatenate the dat files of several Intan sub-recordings into a single
merged session, following Process_ConcatenateDatFiles.m (Peyrache lab).

The sub-recordings are the sub-folders of the session folder, sorted by name
(i.e. by time, given the Intan YYMMDD_HHMMSS naming). For each file type,
the binary content is simply appended in that order.

Outputs written in the session folder:
    <basename>.dat        : concatenated amplifier.dat
    auxiliary.dat, analogin.dat, ...  : other concatenated file types
    <basename>.xml        : copy of the first sub-recording xml
    Epoch_TS.csv          : start/end (seconds) of each sub-recording
    <basename>.cat.evt    : Neuroscope events marking the sub-session limits

Usage:
    python merge_dat_file.py
    python merge_dat_file.py /path/to/session --files amplifier auxiliary analogin
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

CHUNK = 256 * 1024 * 1024  # 256 MB

# file type -> (number of channels, bytes per sample)
# amplifier/auxiliary/analogin/supply are int16, time.dat is int32
DTYPE_SIZE = {"time": 4}


def read_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    n_channels = int(root.find("acquisitionSystem/nChannels").text)
    fs = float(root.find("acquisitionSystem/samplingRate").text)
    n_bits = int(root.find("acquisitionSystem/nBits").text)
    return n_channels, fs, n_bits


def n_samples(path, n_channels, itemsize):
    size = os.path.getsize(path)
    step = n_channels * itemsize
    if size % step:
        raise ValueError(
            "%s : size (%d bytes) is not a multiple of %d channels x %d bytes"
            % (path, size, n_channels, itemsize)
        )
    return size // step


def concatenate(sources, target):
    """Binary append of sources into target, by chunks."""
    total = sum(os.path.getsize(f) for f in sources)
    done = 0
    with open(target, "wb") as fout:
        for src in sources:
            print("    <- %s (%.2f GB)" % (src, os.path.getsize(src) / 1e9))
            with open(src, "rb") as fin:
                while True:
                    buf = fin.read(CHUNK)
                    if not buf:
                        break
                    fout.write(buf)
                    done += len(buf)
                    print("       %.1f %%" % (100 * done / total), end="\r")
    print("    -> %s (%.2f GB)      " % (target, os.path.getsize(target) / 1e9))
    if os.path.getsize(target) != total:
        raise IOError("Size mismatch for %s" % target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session",
        nargs="?",
        default="/Users/gviejo/Downloads/Shan_experiment/B6402-260819",
        help="session folder containing the sub-recording folders",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["amplifier", "auxiliary", "analogin", "supply", "time"],
        help="file types to concatenate (without the .dat extension)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="basename of the merged files (default : name of the session folder)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only print what would be done"
    )
    args = parser.parse_args()

    session = os.path.abspath(args.session)
    basename = args.basename if args.basename else os.path.basename(session)

    # ---------------------------------------------------------------------
    # Sub-recordings, sorted by name (= chronological for Intan folders)
    # ---------------------------------------------------------------------
    subs = sorted(
        d
        for d in os.listdir(session)
        if os.path.isdir(os.path.join(session, d))
        and os.path.exists(os.path.join(session, d, "amplifier.dat"))
    )
    if len(subs) == 0:
        sys.exit("No sub-recording with an amplifier.dat found in %s" % session)

    print("Session  : %s" % session)
    print("Basename : %s" % basename)
    print("Order of concatenation :")
    for i, s in enumerate(subs):
        print("  %d. %s" % (i + 1, s))

    # ---------------------------------------------------------------------
    # Check the recording parameters are identical
    # ---------------------------------------------------------------------
    params = []
    for s in subs:
        xml_file = os.path.join(session, s, "amplifier.xml")
        if not os.path.exists(xml_file):
            sys.exit("Missing %s" % xml_file)
        params.append(read_xml(xml_file))
    if len(set(params)) > 1:
        for s, p in zip(subs, params):
            print("  %s : nChannels=%d, samplingRate=%g, nBits=%d" % ((s,) + p))
        sys.exit("Recording parameters differ between sub-recordings.")

    n_channels, fs, n_bits = params[0]
    print("nChannels = %d, samplingRate = %g Hz, nBits = %d" % (n_channels, fs, n_bits))

    # ---------------------------------------------------------------------
    # Number of samples per sub-recording + consistency across file types
    # ---------------------------------------------------------------------
    durations = []  # in samples, from amplifier.dat
    for s in subs:
        path = os.path.join(session, s, "amplifier.dat")
        durations.append(n_samples(path, n_channels, 2))

    file_types = []
    for ftype in args.files:
        paths = [os.path.join(session, s, ftype + ".dat") for s in subs]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print("Skipping %s.dat (missing in %d sub-recording(s))" % (ftype, len(missing)))
            continue
        itemsize = DTYPE_SIZE.get(ftype, 2)
        # number of channels of that file type, inferred from the first recording
        size = os.path.getsize(paths[0])
        nch, rest = divmod(size, durations[0] * itemsize)
        if rest or nch == 0:
            sys.exit(
                "%s : size not consistent with %d samples of %d bytes"
                % (paths[0], durations[0], itemsize)
            )
        for p, d in zip(paths, durations):
            if n_samples(p, nch, itemsize) != d:
                sys.exit(
                    "%s has %d samples but amplifier.dat has %d"
                    % (p, n_samples(p, nch, itemsize), d)
                )
        file_types.append((ftype, nch, itemsize, paths))
        print("%s.dat : %d channel(s), %d bytes/sample" % (ftype, nch, itemsize))

    total_dur = sum(durations)
    print("Total duration : %d samples = %.2f s = %.2f min"
          % (total_dur, total_dur / fs, total_dur / fs / 60))

    if args.dry_run:
        return

    # ---------------------------------------------------------------------
    # Concatenation
    # ---------------------------------------------------------------------
    for ftype, nch, itemsize, paths in file_types:
        out = basename + ".dat" if ftype == "amplifier" else ftype + ".dat"
        target = os.path.join(session, out)
        if os.path.exists(target):
            sys.exit("%s already exists, remove it first." % target)
        print("Concatenating %s.dat :" % ftype)
        concatenate(paths, target)

    # ---------------------------------------------------------------------
    # xml + epochs
    # ---------------------------------------------------------------------
    xml_target = os.path.join(session, basename + ".xml")
    if not os.path.exists(xml_target):
        shutil.copyfile(os.path.join(session, subs[0], "amplifier.xml"), xml_target)
        print("Copied xml to %s" % xml_target)

    starts, ends, t = [], [], 0
    for d in durations:
        starts.append(t / fs)
        t += d
        ends.append(t / fs)

    epoch_file = os.path.join(session, "Epoch_TS.csv")
    with open(epoch_file, "w") as f:
        for s, e in zip(starts, ends):
            f.write("%.6f,%.6f\n" % (s, e))
    print("Wrote %s" % epoch_file)

    evt_file = os.path.join(session, basename + ".cat.evt")
    with open(evt_file, "w") as f:
        for sub, s, e in zip(subs, starts, ends):
            f.write("%f beginning of %s\n" % (s * 1000, sub))
            f.write("%f end of %s\n" % (e * 1000, sub))
    print("Wrote %s" % evt_file)


if __name__ == "__main__":
    main()
