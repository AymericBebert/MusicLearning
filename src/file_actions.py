#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
This module is used to read and write sound files
Structure of a sound object : sound[canal][sample] (type int list list)
"""

import os
import wave
from subprocess import call
import glob

import numpy as np
from pydub import AudioSegment

from log import writeLog
from timer import timer_start, timer_stop


## Fonctions de conversion entre les fichiers wav et les sounds (listes) ##

def display_params(soundParams):
    """Display some sound metadata"""
    nbch, ss, sf, nbs = soundParams[:4]
    writeLog("info", "Number of channels: {}".format(nbch), {"inFile": False})
    writeLog("info", "Sample size in bytes: {}".format(ss), {"inFile": False})
    writeLog("info", "Sampling frequency: {}".format(sf), {"inFile": False})
    writeLog("info", "Number of samples: {}".format(nbs), {"inFile": False})
    writeLog("info", "Duration : {:.1f}s".format(nbs/sf), {"inFile": False})


def extract_sound(soundFileLoc):
    """Extract the wav file soundFileLoc into a numpy array, shape (nbch, nbs)"""
    timer_start("Extracting {}".format(soundFileLoc))
    err_return = np.array([[]]), (0, 0, 0, 0, "NONE", "not compressed")
    if os.path.splitext(soundFileLoc)[1].lower() in (".mp3", ".wav", ".au"):
        try:
            audiofile = AudioSegment.from_file(soundFileLoc)
        except Exception:
            writeLog("error", "File not found or other I/O error. (DECODING FAILED)")
            return err_return

        if audiofile.sample_width == 2:
            data = np.fromstring(audiofile.raw_data, np.int16)
        elif audiofile.sample_width == 4:
            data = np.fromstring(audiofile.raw_data, np.int32)
        else:
            writeLog("error", "extract_sound(): sample_width is not 2 or 4")
            return err_return
        sf = audiofile.frame_rate
        x = []
        for chn in range(audiofile.channels):
            x.append(data[chn::audiofile.channels])
        x = np.array(x)
    else:
        writeLog("error", "readAudioFile(): Unknown file type!")
        return err_return
    timer_stop("Extracting {}".format(soundFileLoc))
    # number of channels, sample size, sampling frequency, number of samples
    return x, (x.shape[0], audiofile.sample_width, sf, x.shape[1], "NONE", "not compressed")


def write_sound(sound, soundParams, soundFileLoc):
    """Write sound in wav file soundFileLoc, croping values (saturation)"""
    timer_start("Writing {}".format(soundFileLoc))
    nbch, ss, _, nbs = soundParams[:4]
    if nbch == 1:
        rawSound = sound[0]
    elif nbch == 2:
        rawSound = sound.T.reshape((1, 2*sound.shape[1]))[0, :]
    else:
        writeLog("error", "Not a mono nor stereo file")
    soundFile = wave.open(soundFileLoc, "w")
    soundFile.setparams(soundParams)
    # writing binary of nbch*nbs int, taken from rawSound
    soundFile.writeframes(wave.struct.pack("<"+str(nbch*nbs)+("-bhiq"[ss]), *rawSound))
    soundFile.close()
    timer_stop("Writing {}".format(soundFileLoc))
    print("Fichier {} created!".format(soundFileLoc))


### Function working on sounds ###

def empty_sound(nbch, nbs, dtype=np.int16):
    """Generates an empty sound"""
    return np.zeros((nbch, nbs), dtype=dtype)


def convert_to_mono(sound):
    """Converts a stereo sound to mono"""
    nbch = len(sound)
    if nbch == 1:
        return sound
    elif nbch == 2:
        return np.array([(sound[0]/2 + sound[1]/2)]).astype(sound.dtype)
    else:
        writeLog("error", "Not a mono nor stereo file")


def convert_to_stereo(sound):
    """Converts a mono sound to stereo"""
    nbch = len(sound)
    if nbch == 2:
        return sound
    elif nbch == 1:
        return np.array([sound[0], sound[0]])
    else:
        writeLog("error", "Not a mono nor stereo file")


### Fonction de conversion wav <-> mp3 et d'exploitation directe de mp3 ###

def to_mp3(ini, out, bitrate="320k"):
    """Converts the file to an mp3 file"""
    options = ["-threads", "auto", "-y", "-loglevel", "quiet"]
    call(["avconv", "-i", ini, "-c:a", "libmp3lame", "-ab", bitrate] + options + [out])
    writeLog("debug", "File {} created!".format(out))


def to_wav(ini, out):
    """Converts the file to an wav file"""
    # options = ["-threads", "auto", "-y", "-loglevel", "quiet"]
    # call(["avconv", "-i", ini] + options + [out])
    call(["mpg123", "-w", out, ini])
    writeLog("debug", "File {} created!".format(out))


def folder_mp3_to_wav():
    """Convert all not already converted mp3 to wav"""
    for _, f in enumerate(os.listdir("../data/samplesMP3")):
        samplesMP3 = glob.glob("../data/samplesMP3/{}/*.mp3".format(f))
        for s in samplesMP3:
            trackName = os.path.splitext(os.path.split(s)[1])[0]
            swav = "../data/samples/{}/{}.wav".format(f, trackName)
            if not os.path.isdir(os.path.split(swav)[0]):
                os.mkdir(os.path.split(swav)[0])
            if not os.path.isfile(swav):
                to_wav(s, swav)
    writeLog("debug", "MP3 to WAV conversions finished.")


def mp3_mp3(fonction, soundFileLoc1, soundFileLoc2, moreArgs=None, bitrate="320k"):
    """Wraps a wav -> wav function to make it mp3 -> mp3"""
    if moreArgs is None:
        moreArgs = []
    to_wav(soundFileLoc1, "./temp1.wav")
    fonction("./temp1.wav", "./temp2.wav", *moreArgs)
    to_mp3("./temp2.wav", soundFileLoc2, bitrate)
    os.remove("./temp1.wav")
    os.remove("./temp2.wav")
