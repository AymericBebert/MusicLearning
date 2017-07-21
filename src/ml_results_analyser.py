# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
This module can display ML results
"""

from log import writeLog


def fillIfNeeded(A, B):
    """Fill A or B with zeros if not the same size"""
    A = list(A)[:]
    B = list(B)[:]
    if len(A) < len(B):
        A += [0]*(len(B)-len(A))
    if len(B) < len(A):
        B += [0]*(len(A)-len(B))
    return A, B


def printResults(labels, output):
    """Logs the results"""
    writeLog("debug", "output: {}".format(", ".join([str(x).zfill(2) for x in output])))
    writeLog("debug", "labels: {}".format(", ".join([str(y).zfill(2) for y in labels])))
    labels, output = fillIfNeeded(labels, output)
    confusionMatrix(labels, output)
    perfsByLabel(labels, output)


def confusionMatrix(Y, Z):
    """Print a crude confusion matrix"""
    M = []
    n = max((Y)) + 1
    for _ in range(n):
        M.append([0]*n)
    for i, y in enumerate(Y):
        M[y][Z[i]] += 1
    s = "Confusion matrix:\n"
    s += "L\\O" + "|".join([str(x).rjust(2) for x in range(n)]) + "\n"
    for i in range(n):
        s += str(i).zfill(2) + "|" + " ".join([str(x).rjust(2) for x in M[i]]) + "\n"
    print(s)


def perfsByLabel(Y, Z):
    """Displays the performances by label (x out of y good, z wrong)"""
    n = max((Y)) + 1
    good = [0] * n
    bad = [0] * n
    count = [0] * n
    for i, y in enumerate(Y):
        count[y] += 1
        if y == Z[i]:
            good[y] += 1
        else:
            bad[Z[i]] += 1
    s = "Good results by label:\n"
    for i in range(n):
        s += "{: >2}: {}/{} good, {} bad\n".format(i, good[i], count[i], bad[i])
    print(s)
