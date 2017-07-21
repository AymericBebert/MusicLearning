# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
This module is used to log data in the console and into log files
There are several levels of logging, with different colors and files
"""

import os
import sys
import subprocess
from time import strftime, localtime


logConfig = {}

consoleColors = {
    "error": "\033[91m",
    "warn": "\033[93m",
    "info": "\033[96m",
    "access": "\033[0m",
    "debug": "\033[90m",
    "time": "\033[90m",
    "call": "\033[93m",
}


def uprint(*objects, sep=" ", end="\n", file=sys.stdout):
    """Wrapper for print, useful with Windows (avoid some encoding problems)"""
    enc = file.encoding
    if enc == "UTF-8":
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors="backslashreplace").decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)


def fileNameByLevel(level):
    """Returns the filename associated with the service and the log level"""
    fname = "{}-{}.log".format(logConfig["logFilePrefix"], level)
    return os.path.join(logConfig["logFolder"], fname)


def initLog(config):
    """Initialize the config, makes the directory if needed,
       disables default logger"""
    logConfig["logFolder"] = config.get("General", "logFolder")
    logConfig["logFilePrefix"] = config.get("General", "logFilePrefix")
    logConfig["debugLogs"] = config.get("General", "debugLogs").lower() == "true"

    if not os.path.exists(logConfig["logFolder"]):
        os.makedirs(logConfig["logFolder"])


def writeLog(level, text, options=None, limit=3000):
    """Write the text in console and file, depending on the level
       options : set "inConsole" or "inFile" to False to disable this"""
    if options is None:
        options = {}
    if "inConsole" not in options:
        options["inConsole"] = (level != "debug") or logConfig["debugLogs"]
    if "inFile" not in options:
        options["inFile"] = (level != "debug")

    # In console
    if options["inConsole"] is not False:
        time = strftime("%H:%M:%S", localtime())
        s = "{} - {} - {}".format(time, level.upper(), text)
        if limit is not None and len(s) > limit-3:
            s = s[:limit-3] + "..."
        if level in consoleColors:
            s = consoleColors[level] + s + "\033[0m"
        uprint(s)

    # In file
    if options["inFile"] is not False:
        time = strftime("%Y-%m-%d %H:%M:%S", localtime())
        filename = "{}-{}.log".format(logConfig["logFilePrefix"], level)
        f = open(os.path.join(logConfig["logFolder"], filename), "a")
        f.write("{} - {}\n".format(time, text))
        f.close()


def logAndCall(arg):
    """Log the command and call it"""
    if isinstance(arg, str):
        s = arg
    else:
        s = " ".join(['"{}"'.format(a) if " " in a else a for a in arg])
    writeLog("call", s, {"inFile": False})
    subprocess.call(arg)
