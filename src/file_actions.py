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
    """Extract the wav file soundFileLoc into a list of integers"""
    timer_start("Extracting {}".format(soundFileLoc), True)
    soundFile = wave.open(soundFileLoc, "rb")
    soundParams = soundFile.getparams()
    # number of channels, sample size, sampling frequency, number of samples
    nbch, ss, _, nbs = soundParams[:4]
    param = "-bhiq"[ss]
    if nbch == 1:
        sound = [wave.struct.unpack("<"+str(nbs)+param, soundFile.readframes(nbs))]
    elif nbch == 2:
        sound0 = wave.struct.unpack("<"+str(2*nbs)+param, soundFile.readframes(nbs))
        sound = [[0]*nbs, [0]*nbs]
        for i in range(nbs):
            sound[0][i], sound[1][i] = sound0[2*i], sound0[2*i+1]
    else:
        writeLog("error", "Not a mono nor stereo file")
    soundFile.close()
    timer_stop("Extracting {}".format(soundFileLoc))
    return sound, soundParams


def extract_sound_light(soundFileLoc, ratio=0.3, duration=5):
    """
    Extract a part of the wav file soundFileLoc into a list of integers
    ratio: where to start (0.-1.), duration: number of seconds to extract
    """
    timer_start("Extracting {}".format(soundFileLoc), True)
    soundFile = wave.open(soundFileLoc, "rb")
    soundParams = soundFile.getparams()
    # number of channels, sample size, sampling frequency, number of samples
    nbch, ss, sf, nbs = soundParams[:4]
    param = "-bhiq"[ss]
    if nbch == 1:
        soundFile.setpos(int(nbs*ratio))
        nbs_light = duration*sf
        sound = [wave.struct.unpack("<"+str(nbs_light)+param, soundFile.readframes(nbs_light))]
    elif nbch == 2:
        soundFile.setpos(2*(int(nbs*ratio)//2))
        nbs_light = 2*((duration*sf)//2)
        sound0 = wave.struct.unpack("<"+str(2*nbs_light)+param, soundFile.readframes(nbs_light))
        # directly transform to mono
        sound = [[x for i, x in enumerate(sound0) if i%2 == 0]]
    else:
        writeLog("error", "Not a mono nor stereo file")
    soundFile.close()
    timer_stop("Extracting {}".format(soundFileLoc))
    soundParams = (1,) + soundParams[1:3] + (nbs_light,) + soundParams[4:]
    return sound, soundParams


def crop(x, a, b):
    """crop x between a and b"""
    if x < a:
        return a
    elif x > b:
        return b
    return x


def write_sound(sound, soundParams, soundFileLoc):
    """Write sound in wav file soundFileLoc, croping values (saturation)"""
    timer_start("Writing {}".format(soundFileLoc), True)
    nbch, ss, _, nbs = soundParams[:4]
    if nbch == 1:
        rawSound = sound[0]
    elif nbch == 2:
        rawSound = [0]*(2*nbs)
        for i in range(nbs):
            rawSound[2*i], rawSound[2*i+1] = sound[0][i], sound[1][i]
    else:
        writeLog("error", "Not a mono nor stereo file")
    rawSound = [crop(e, -32768, 32767) for e in rawSound]
    soundFile = wave.open(soundFileLoc, "w")
    soundFile.setparams(soundParams)
    # writing binary of nbch*nbs int, taken from rawSound
    soundFile.writeframes(wave.struct.pack("<"+str(nbch*nbs)+("-bhiq"[ss]), *rawSound))
    soundFile.close()
    timer_stop("Writing {}".format(soundFileLoc))
    print("Fichier {} created!".format(soundFileLoc))


### Function working on sounds ###

def empty_sound(nbch, nbs):
    """Generates an empty sound"""
    return [[0]*nbs for c in range(nbch)]


def convert_to_mono(sound):
    """Converts a stereo sound to mono"""
    nbch = len(sound)
    if nbch == 1:
        return sound
    elif nbch == 2:
        return [[(v1+v2)//2 for v1, v2 in zip(sound[0], sound[1])]]
    else:
        writeLog("error", "Not a mono nor stereo file")


def convert_to_stereo(sound):
    """Converts a mono sound to stereo"""
    nbch = len(sound)
    if nbch == 2:
        return sound
    elif nbch == 1:
        return [sound[0], sound[0]]
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
