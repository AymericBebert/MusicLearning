# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""
This module is used to time and log parts of code
"""

from time import perf_counter

from log import writeLog

time_data = {}

def timer_start(category, verbose=False):
    """Starts the timer"""
    if verbose:
        writeLog("time", 'Timer started for "{}"'.format(category), {"inFile": False})
    time_data[category] = perf_counter()

def timer_stop(category):
    """Stops the timer, log and returns the time in seconds"""
    try:
        dt = perf_counter() - time_data[category]
        writeLog("time", '{:.03f}s for "{}"'.format(dt, category), {"inFile": False})
        return dt
    except KeyError:
        writeLog("warn", 'No time data for "{}"'.format(category))
        return 0
