#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
This module is used to extract features from the data
"""

import numpy as np
import python_speech_features


def file_length(soundParams):
    """Returns the file length, in seconds"""
    return soundParams[3] / soundParams[2]


def extract_mfcc(sound_obj, avg=True):
    """Extract the MFCC from the sound object"""
    soundD = sound_obj["sound"]  # raw data
    sr = sound_obj["params"][2]  # samplerate
    # nf = sound_obj["params"][3]  # nframes

    test_mfcc = python_speech_features.mfcc(soundD, samplerate=sr, winlen=0.025, winstep=1)
    if avg:
        return np.mean(test_mfcc, axis=0)
    return test_mfcc

def extract_all_features(sound_obj):
    """Extract the features from the sound object"""
    # fl = file_length(sound_obj["params"])
    test_mfcc_avg = extract_mfcc(sound_obj)
    # return np.concatenate(([fl], test_mfcc_avg))
    return test_mfcc_avg

def features_labels():
    """Give a name to each feature"""
    return ["mfcc{}".format(i) for i in range(13)]
