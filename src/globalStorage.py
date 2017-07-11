# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
This singleton-pattern module can be used to store and edit global variables
across multiple files
"""


class GlobalStorage:
    """Singleton class for storage"""

    class __GlobalStorage:
        def __init__(self):
            self.nope = None

        def __str__(self):
            return repr(self) + self.nope

        def getNope(self):
            """Gets the nope"""
            return self.nope

        def setNope(self, nope):
            """Sets the nope"""
            self.nope = nope

    instance = None

    def __init__(self):
        if not GlobalStorage.instance:
            GlobalStorage.instance = GlobalStorage.__GlobalStorage()

    def __getattr__(self, name):
        return getattr(self.instance, name)
