#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
This module is used to extract features from the data
"""

import numpy as np
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import python_speech_features

eps = 0.00000001


def file_length(soundParams):
    """Returns the file length, in seconds"""
    return soundParams[3] / soundParams[2]


def zcr(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return countZ / (count - 1)


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / len(frame)


def energy_entropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    tfe = np.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(np.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
        frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(subWindows ** 2, axis=0) / (tfe + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -1 * np.sum(s * np.log2(s + eps))
    return entropy


def spectral_centroid_and_spread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    C = (NUM / DEN)  # Centroid
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)  # Spread

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def avg_mfcc(sound_obj, avg=True):
    """Extract the MFCC from the sound object"""
    soundD = sound_obj["sound"]  # raw data
    sr = sound_obj["params"][2]  # samplerate
    # nf = sound_obj["params"][3]  # nframes

    all_mfcc = python_speech_features.mfcc(soundD, samplerate=sr, winlen=0.025, winstep=1)
    if avg:
        return np.mean(all_mfcc, axis=0)
    return all_mfcc


def mfcc_init_filter_banks(fs, nfft):
    """Computes the triangular filterbank for MFCC computation"""

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1, np.floor(cenTrFreq * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1, np.floor(highTrFreq * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def mfcc(X, fbank, nceps=13):
    """Computes the MFCCs of a frame, given the fft mag"""
    mspec = np.log10(np.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def extract_all_features0(sound_obj):
    """Extract the features from the sound object"""
    # fl = file_length(sound_obj["params"])
    test_mfcc_avg = avg_mfcc(sound_obj)
    # return np.concatenate(([fl], test_mfcc_avg))
    return test_mfcc_avg


def features_labels0():
    """Give a name to each feature"""
    return ["mfcc{}".format(i) for i in range(13)]


def extract_all_features(sound_obj, wins=None, steps=None):
    """Extract the features from the sound object"""
    sr = sound_obj["params"][2]  # samplerate
    nbs = sound_obj["params"][3]  # number of samples
    if wins is None:
        wins = int(0.050 * sr)
    if steps is None:
        steps = int(nbs/15 - wins)

    # Signal normalization
    signal = sound_obj["sound"]
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)      # total number of samples
    curPos = steps // 2  # skip the very beginning
    nFFT = wins // 2

    # compute the triangular filter banks used in the mfcc calculation
    #[fbank, _] = mfcc_init_filter_banks(sr, nFFT)

    totalNumOfFeatures = 5 + 13

    stFeatures = []
    while curPos + wins - 1 < N:  # for each short-term window until the end of signal
        x = signal[curPos:curPos+wins]                   # get current window
        curPos = curPos + steps                          # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        curFV = np.zeros(totalNumOfFeatures)
        curFV[0] = zcr(x)                                # zero crossing rate
        curFV[1] = energy(x)                             # short-term energy
        curFV[2] = energy_entropy(x)                     # short-term entropy of energy
        [curFV[3], curFV[4]] = spectral_centroid_and_spread(X, sr)    # spectral centroid and spread
        # curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        # curFV[6] = stSpectralFlux(X, Xprev)              # spectral flux
        # curFV[7] = stSpectralRollOff(X, 0.90, sr)        # spectral rolloff
        # curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = stMFCC(X, fbank, nceps).copy()    # MFCCs
        #
        # chromaNames, chromaF = stChromaFeatures(X, sr, nChroma, nFreqsPerChroma)
        # curFV[numOfTimeSpectralFeatures + nceps: numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF
        # curFV[numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF.std()

        #curFV[5:18] = mfcc(X, fbank, 13)
        #curFV[0:13] = mfcc(X, fbank, 13)
        curFV[5:18] = python_speech_features.mfcc(x, samplerate=sr, winlen=wins/sr, winstep=steps/sr)

        # TEMP
        #curFV = python_speech_features.mfcc(signal, samplerate=sr, winlen=wins, winstep=steps).T

        stFeatures.append(curFV)

    # stFeatures = np.array(stFeatures)
    stFeatures = np.concatenate(stFeatures, 0).flatten()
    #stFeatures = np.mean(stFeatures, axis=0)
    # stFeatures = python_speech_features.mfcc(signal, samplerate=sr, winlen=wins/sr, winstep=steps/sr)
    # stFeatures = np.mean(stFeatures, axis=0)
    return stFeatures

    # sound_obj2 = sound_obj.copy()
    # sound_obj2["sound"] = signal
    #
    # # fl = file_length(sound_obj["params"])
    # test_mfcc_avg = avg_mfcc(sound_obj2)
    # # return np.concatenate(([fl], test_mfcc_avg))
    # return test_mfcc_avg


def features_labels():
    """Give a name to each feature"""
    return ["zrc", "energy", "en_ent", "centr", "spread"] + ["mfcc{}".format(i) for i in range(13)]
