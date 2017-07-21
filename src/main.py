#!/usr/bin/env python3
# -*-coding:utf-8-*-

"""
Main module, still mostly tests
"""

import os
import sys
import glob
import random
import itertools

from configparser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

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
writeLog("info", "Folders in ../data/samples: {}".format(folders))

file_actions.folder_mp3_to_wav()

folders = os.listdir("../data/samples")
labels = [f.title() for f in folders]
writeLog("info", "Folders in ../data/samples: {}".format(folders))

def recover_saved_data():
    """Recover previously saved data"""
    X = np.loadtxt("../tmp/X.csv")
    Y = np.loadtxt("../tmp/Y.csv").astype("int")
    with open("../tmp/flabels.txt", "r") as f:
        flabels = [l.strip() for l in f.readlines()]
    with open("../tmp/trackNames.txt", "r") as f:
        trackNames = [l.strip() for l in f.readlines()]
    return X, Y, flabels, trackNames


def file_name_to_track_name(fn):
    """Make a clean track name with the file name"""
    tn = os.path.split(fn)[1]
    tn = os.path.splitext(tn)[0]
    return tn.lstrip('0123456789').lstrip(' -.')

def create_data_from_files():
    """Compute the data (features, labels,...) from audio files and save the results"""
    X = []
    Y = []
    trackNames = []
    # Extract all the features
    for i, f in enumerate(folders):
        samples = glob.glob("../data/samples/{}/*.wav".format(f))
        for s in samples:
            # es = file_actions.extract_sound_light(s, ratio=0.5, duration=10) # test light
            es = file_actions.extract_sound(s)
            esm = np.array(file_actions.convert_to_mono(es[0])[0])
            d = {"label": i, "sound": esm, "params": es[1], "file": s}
            X.append(extract_features.extract_all_features(d))
            Y.append(d["label"])
            trackNames.append(file_name_to_track_name(d["file"]))
    # Make arrays fron the data
    X = np.array(X)
    Y = np.array(Y)
    flabels = extract_features.features_labels()
    # Save the data and returning it
    np.savetxt("../tmp/X.csv", X)
    np.savetxt("../tmp/Y.csv", Y)
    with open("../tmp/flabels.txt", "w") as f:
        f.write("\n".join(flabels))
    with open("../tmp/trackNames.txt", "w") as f:
        f.write("\n".join(trackNames))
    writeLog("info", "File extraction finished")
    return X, Y, flabels, trackNames


load_saved = config.get("ML", "loadSaved").lower() == "true"

if load_saved:
    try:
        X, Y, flabels, trackNames = recover_saved_data()
    except Exception:
        writeLog("warn", "Could not load the data, will extract from files.")
        X, Y, flabels, trackNames = create_data_from_files()
else:
    X, Y, flabels, trackNames = create_data_from_files()

# print(X[:, :4])
# print(Y)
# print(flabels)


### Prepare the data ###

# Shuffle all the samples
data_group = list(zip(X, Y, trackNames))
random.shuffle(data_group)
X, Y, trackNames = list(zip(*data_group))
X = np.array(X)
Y = np.array(Y)
trackNames = np.array(trackNames)

# Normalize and/or scale the data
X_s = preprocessing.scale(X)
X_n = preprocessing.normalize(X)
X_sn = preprocessing.normalize(X_s)


## Cut the data into training and test samples ##

tr_ratio = float(config.get("ML", "trainingRatio"))
sep_ind = int(tr_ratio*len(Y))

# Training samples
X_tr = X[:sep_ind, :]
X_tr_s = X_s[:sep_ind, :]
X_tr_n = X_n[:sep_ind, :]
X_tr_sn = X_sn[:sep_ind, :]
Y_tr = Y[:sep_ind]
trackNames_tr = trackNames[:sep_ind]

# Test samples
X_te = X[sep_ind:, :]
X_te_s = X_s[sep_ind:, :]
X_te_n = X_n[sep_ind:, :]
X_te_sn = X_sn[sep_ind:, :]
Y_te = Y[sep_ind:]
trackNames_te = trackNames[sep_ind:]


def plot_feature_per_label():
    """Plot all features, different colors for label"""
    fig = plt.figure(figsize=(12, 2*X.shape[1])) # (width, height)

    for idx in range(X.shape[1]):
        for (number, Xi, legend) in [(1, X, '(not scaled)'), (2, X_s, '(scaled)')]:
            fig.add_subplot(X.shape[1], 2, 2*idx+number)

            for i in range(len(labels)):
                indexes = [ind for ind in range(len(Y)) if Y[ind] == i]
                xdata = range(len(indexes))
                ydata = [Xi[ind, idx] for ind in indexes]

                plt.plot(xdata, ydata, 'o')
            plt.title("{} {}".format(flabels[idx], legend), fontsize=16)

    plt.tight_layout() # improve spacing between subplots

# plot_feature_per_label()


### Training ###

def train_and_fit(classifier, X_tr, Y_tr, X_te):
    """Train the classifier using X_tr and Y_tr, and fit X_te"""
    classifier.fit(X_tr, Y_tr)
    return classifier.predict(X_te).astype('int')


# Set up a stratified 3-fold cross-validation
folds = model_selection.StratifiedKFold(3, shuffle=True)

def cross_validate(classifier, design_matrix, labels, cv_folds):
    """Perform a cross-validation and returns the predictions."""
    pred = np.zeros(labels.shape)
    for tr, te in cv_folds.split(design_matrix, labels):
        # Restrict data to train/test folds
        Xtr = design_matrix[tr, :]
        ytr = labels[tr]
        Xte = design_matrix[te, :]

        # Fit classifier
        classifier.fit(Xtr, ytr)

        # Predict the label with the features
        yte_pred = classifier.predict(Xte)
        pred[te] = yte_pred[:]
    return pred.astype('int')


## Results visualization ##

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def analyse_results(Ypred, Y, trackNames=trackNames, details=1):
    """
    Analyse the results, several levels of details:
    0: Print the score
    1: Print the score and confusion matrix
    2: Print the score, confusion matrix, and prediction for each track
    """
    print("Labels:    ", Y)
    print("Prediction:", Ypred)

    score = sum([1 if Ypred[i] == yi else 0 for i, yi in enumerate(Y)])
    ratio = score / len(Y)
    writeLog("info", "Score: {:03f}  ({}/{})".format(ratio, score, len(Y)))

    if details >= 1:
        cnf_matrix = confusion_matrix(Y, Ypred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=labels, title='Confusion matrix')
        plt.show()
    if details >= 2:
        for i, track in enumerate(trackNames):
            if Ypred[i] != Y[i]:
                res = "{} -> {} ({})".format(track, labels[Ypred[i]], labels[Y[i]])
                res = "\033[91m{}\033[0m".format(res)
            else:
                res = "{} -> {}".format(track, labels[Y[i]])
            print(res)


## Logistic regression

# clf_lr = linear_model.LogisticRegression(C=1e6) # high C means no regularization

# Ypred_lr = train_and_fit(clf_lr, X_tr, Y_tr, X_te)
# analyse_results(Ypred_lr, Y_te, trackNames_te)

# Ypred_lr = cross_validate(clf_lr, X, Y, folds)
# analyse_results(Ypred_lr, Y, trackNames)


## Logistic regression, scaled data

clf_lr_s = linear_model.LogisticRegression(C=1e6) # high C means no regularization

# Ypred_lr_s = train_and_fit(clf_lr_s, X_tr_s, Y_tr, X_te_s)
# analyse_results(Ypred_lr_s, Y_te, trackNames_te, details=2)

Ypred_lr_s = cross_validate(clf_lr_s, X_s, Y, folds)
analyse_results(Ypred_lr_s, Y, trackNames, details=2)
