#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
Main module
"""

import os
import sys
from configparser import ConfigParser

from globalStorage import GlobalStorage
from log import initLog, writeLog

# Reading the config file
config = ConfigParser()
cfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini")
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
