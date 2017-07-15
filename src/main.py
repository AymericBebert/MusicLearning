#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
Main module, still mostly tests
"""

import os
import sys
import glob

from configparser import ConfigParser
import numpy as np

from globalStorage import GlobalStorage
from log import initLog, writeLog
import file_actions
import extract_features


# Reading the config file
config = ConfigParser()
cfile = os.path.join(os.getcwd(), "config.ini")
config.read(cfile)

# Logs initialization
initLog(config)
writeLog("info", "Program restarted")

# Python version info
sys.path.append(os.path.abspath(os.getcwd()))
python_version = sys.version_info.major
writeLog("debug", "Python version: {}".format(sys.version))

# Instanciationg the global storage
gs = GlobalStorage()

folders = os.listdir("../data/samples")
writeLog('info', "folders: {}".format(folders))

for i, f in enumerate(folders):
    samplesMP3 = glob.glob("../data/samples/{}/*.mp3".format(f))
    for s in samplesMP3:
        swav = os.path.splitext(s)[0] + '.wav'
        if not os.path.isfile(swav):
            file_actions.to_wav(s, swav)

data = []
for i, f in enumerate(folders):
    samples = glob.glob("../data/samples/{}/*.wav".format(f))
    for s in samples:
        es = file_actions.extract_sound(s)
        esm = np.array(file_actions.convert_to_mono(es[0])[0])
        data.append({"label": i, "sound": esm, "params": es[1], "file": s})


# print("{} <- {}".format(data[0]["label"], data[0]["file"]))
# print(data[0]["sound"][100000:100010])
# print(data[0]["params"])
# print(extract_features.file_length(data[0]["params"]))


test_features = extract_features.extract_all_features(data[0])
print(test_features)
