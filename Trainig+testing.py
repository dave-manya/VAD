# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 18:42:18 2018

@author: hp
"""

import tensorflow as tf
with tf.device("/GPU:0"):
    
    import numpy as np
    import os
    import pickle
    import h5py
    import argparse
    import time
    import glob
    import matplotlib.pyplot as plt
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.optimizers import Adam
    from keras.models import load_model
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed
    from keras.layers import Activation, CuDNNLSTM
    from keras import optimizers
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    import scipy.io as spio
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    

    # MODEL - I: MFCC

    # Defining variables
    batch_size = 128
    data_dim = 39
    #timesteps = ___
    lr = 0.001
        
    # Building Network
    print("Building network...")
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Dense(64, input_shape=(None, data_dim)))
    model.add((CuDNNLSTM(128, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add((Dense(1, activation='linear')))
    
    print("Learning rate = ", lr)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    model.summary()
    

    # Loading .mat files and storing data in .npy form
    print("Loading data...")
    x_train = spio.loadmat('Path to .mat file')
    x_test = spio.loadmat('Path to .mat file')
    y_train = spio.loadmat('Path to .mat file')
    x_val = spio.loadmat('Path to .mat file')
    y_test = spio.loadmat('Path to .mat file')
    y_val = spio.loadmat('Path to .mat file')
    x_train = x_train["train_feat"]
    y_train = y_train["train_label"]
    x_train = x_train.reshape((1, 547978, 39))
    y_train = y_train.reshape((1, 547978, 1))
    x_test = x_test["test_feat"]
    y_test = y_test["test_label"]
    x_test = x_test.reshape((1, 15282, 39))
    y_test = y_test.reshape((1, 15282, 1))
    x_val = x_val["train_feat"]
    y_val = y_val["train_label"]
    x_val = x_val.reshape((1, 22027, 39))
    y_val = y_val.reshape((1, 22027, 1))
    
    np.save('x_train_MFCC_500.npy', x_train)
    np.save('x_test_MFCC.npy', x_test)
    np.save('y_train_500.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('x_val_MFCC', x_val)
    np.save('y_val', y_val)
    
    x_train = np.load('Path to .npy file')
    x_test = np.load('Path to .npy file')
    y_train = np.load('Path to .npy file')
    y_test = np.load('Path to .npy file')
    x_val = np.load('Path to .npy file')
    y_val = np.load('Path to .npy file')


    # Training and saving the model
    print("Training model...")        
    history = model.fit(x_train, y_train, batch_size, epochs=100, validation_data=(x_val, y_val) ,shuffle=True)
    
    model.save('Model_MFCC.h5')
    

    # Testing and predicting the model
    print("Testing model...")
    score = model.evaluate(x_test, y_test, batch_size)
    print('\nFinal Score: [Loss, Accuracy] =', score)
    
    y_pred = model.predict(x_test).ravel()
    print("\nPredicted labels: ",y_pred)
    

    # Plotting accuracy and loss vs epoch    

    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

    

    # Plotting ROC curve and finding AUC after median filtering
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from scipy.signal import medfilt
    y_test = spio.loadmat('Path to .mat file')
    y_test = y_test["test_label"]
    y_test = np.array(y_test.reshape(15282))
    
    print(y_pred)
    y_pred_M = medfilt(y_pred, 11)
    print(y_pred_M)
    
    print('True labels:', y_test)
    print('Predicted labels:',y_pred_M.ravel())
    auc_score = roc_auc_score(y_test, y_pred_M)
    print('AUC score: ', auc_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_M)
    AUC = auc(fpr, tpr)
    
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve-MFCC')
    plt.legend(loc='best')
    plt.show()
#    # Zoom in view of the upper left corner.
#    plt.figure(2)
#    plt.xlim(0, 0.2)
#    plt.ylim(0.8, 1)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.title('ROC curve (zoomed in at top left)')
#    plt.legend(loc='best')
#    plt.show()
#    

    # Finding Confusion matrix and equal error rate
    y = []
    for i in y_pred_M:
        if i > 0.435:
            i = 1
        else:
            i = 0
        y.append(i)
        
    #print(y)    
    
    cm = confusion_matrix(y_test, y)
    tn, fp, fn, tp = confusion_matrix(y_test, y).ravel()
    print('\nConfusion Matrix :\n', cm)
    print('True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d' %(tn, fp, fn, tp))
    
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print("Equal error rate = ", eer)
    

    # MODEL - II: SADJADI     

    # Defining variables
    batch_size = 128
    data_dim = 12
    # timesteps = ___
    lr = [0.001]


    # Building Network
    print("Building network...")
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Dense(64, input_shape=(None, data_dim)))
    model.add((CuDNNLSTM(128, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add((Dense(1, activation='linear')))
    
    print("Learning rate = ", lr)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    model.summary()


    # Loading .mat files and storing data in .npy form
    print("Loading data...")
    x_train = spio.loadmat('Path to .mat file')
    x_test = spio.loadmat('Path to .mat file')
    y_train = spio.loadmat('Path to .mat file')
    x_val = spio.loadmat('Path to .mat file')
    y_test = spio.loadmat('Path to .mat file')
    y_val = spio.loadmat('Path to .mat file')
    x_train = x_train["train_feat"]
    y_train = y_train["train_label"]
    x_train = x_train.reshape((1, 547978, 12))
    y_train = y_train.reshape((1, 547978, 1))
    x_test = x_test["test_feat"]
    y_test = y_test["test_label"]
    x_test = x_test.reshape((1, 15282, 12))
    y_test = y_test.reshape((1, 15282, 1))
    x_val = x_val["train_feat"]
    y_val = y_val["train_label"]
    x_val = x_val.reshape((1, 22027, 12))
    y_val = y_val.reshape((1, 22027, 1))
    
    np.save('x_train_Sadjadi_500.npy', x_train)
    np.save('x_test_Sadjadi.npy', x_test)
    np.save('y_train_500.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('x_val_Sadjadi', x_val)
    np.save('y_val', y_val)
    
    x_train = np.load('Path to .npy file')
    x_test = np.load('Path to .npy file')
    y_train = np.load('Path to .npy file')
    y_test = np.load('Path to .npy file')
    x_val = np.load('Path to .npy file')
    y_val = np.load('Path to .npy file')


    # Training and saving model
    print("Training model...")        
    history = model.fit(x_train, y_train, batch_size, epochs=100, validation_data=(x_val, y_val) ,shuffle=True)
    
    model.save('Model_Sadjadi.h5')
    
    
    # Testing the model and predicting labels
    print("Testing model...")
    score = model.evaluate(x_test, y_test, batch_size)
    print('\nFinal Score: [Loss, Accuracy] =', score)
    
    y_pred = model.predict(x_test).ravel()
    print("\nPredicted labels: ",y_pred)
    

    # Plotting accuracy and loss vs epochs

    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

    # Plotting ROC curve and AUC
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from scipy.signal import medfilt
    y_test = spio.loadmat('Path to .mat file')
    y_test = y_test["test_label"]
    y_test = np.array(y_test.reshape(15282))
    
    print(y_pred)
    y_pred_S = medfilt(y_pred, 11)
    print(y_pred_S)
    
    print('True labels:', y_test)
    print('Predicted labels:',y_pred_S.ravel())
    auc_score = roc_auc_score(y_test, y_pred_S)
    print('AUC score: ', auc_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_S)
    AUC = auc(fpr, tpr)

    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve-Sadjadi')
    plt.legend(loc='best')
    plt.show()
#    # Zoom in view of the upper left corner.
#    plt.figure(2)
#    plt.xlim(0, 0.2)
#    plt.ylim(0.8, 1)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.title('ROC curve (zoomed in at top left)')
#    plt.legend(loc='best')
#    plt.show()
#  
    

    # Finding confusion matrix and equal error rate
    y = []
    for i in y_pred_S:
        if i > 0.435:
            i = 1
        else:
            i = 0
        y.append(i)
        
    #print(y)    
    
    cm = confusion_matrix(y_test, y)
    tn, fp, fn, tp = confusion_matrix(y_test, y).ravel()
    print('\nConfusion Matrix :\n', cm)
    print('True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d' %(tn, fp, fn, tp))
    
  
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print("Equal error rate = ", eer)
        
        
  # MODEL - III: NEW

    # Defining variables
    batch_size = 128
    data_dim = 9
    # timesteps = ___
    lr = [0.001]


    # Building network    
    print("Building network...")
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Dense(64, input_shape=(None, data_dim)))
    model.add((CuDNNLSTM(128, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add((Dense(1, activation='linear')))
    
    print("Learning rate = ", lr)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    model.summary()
    

    # Loading .mat files and storing data in .npy form
    print("Loading data...")
    x_train = spio.loadmat('Path to .mat file')
    x_test = spio.loadmat('Path to .mat file')
    y_train = spio.loadmat('Path to .mat file')
    x_val = spio.loadmat('Path to .mat file')
    y_test = spio.loadmat('Path to .mat file')
    y_val = spio.loadmat('Path to .mat file')
    x_train = x_train["train_feat"]
    y_train = y_train["train_label"]
    x_train = x_train.reshape((1, 547978, 9))
    y_train = y_train.reshape((1, 547978, 1))
    x_test = x_test["test_feat"]
    y_test = y_test["test_label"]
    x_test = x_test.reshape((1, 15282, 9))
    y_test = y_test.reshape((1, 15282, 1))
    x_val = x_val["train_feat"]
    y_val = y_val["train_label"]
    x_val = x_val.reshape((1, 22027, 9))
    y_val = y_val.reshape((1, 22027, 1))
    
    np.save('x_train_New_500.npy', x_train)
    np.save('x_test_New.npy', x_test)
    np.save('y_train_500.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('x_val_New', x_val)
    np.save('y_val', y_val)
    
    x_train = np.load('Path to .npy file')
    x_test = np.load('Path to .npy file')
    y_train = np.load('Path to .npy file')
    y_test = np.load('Path to .npy file')
    x_val = np.load('Path to .npy file')
    y_val = np.load('Path to .npy file')


    # Training and saving the model
    print("Training model...")        
    history = model.fit(x_train, y_train, batch_size, epochs=100, validation_data=(x_val, y_val) ,shuffle=True)
    
    model.save('Model_New.h5')


    # Testing the model and finding predicted labels
    print("Testing model...")
    score = model.evaluate(x_test, y_test, batch_size)
    print('\nFinal Score: [Loss, Accuracy] =', score)
    
    y_pred = model.predict(x_test).ravel()
    print("\nPredicted labels: ",y_pred)
    

    # Plotting Accuracy and Loss vs Epochs
    
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

    # Plotting ROC curve and finding AUC score
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from scipy.signal import medfilt
    y_test = spio.loadmat('Path to .mat file')
    y_test = y_test["test_label"]
    y_test = np.array(y_test.reshape(15282))
    
    print(y_pred)
    y_pred_N = medfilt(y_pred, 11)
    print(y_pred_N)
    
    print('True labels:', y_test)
    print('Predicted labels:',y_pred_N.ravel())
    auc_score = roc_auc_score(y_test, y_pred_N)
    print('AUC score: ', auc_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_N)
    AUC = auc(fpr, tpr)
    
      
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve-New')
    plt.legend(loc='best')
    plt.show()
#    # Zoom in view of the upper left corner.
#    plt.figure(2)
#    plt.xlim(0, 0.2)
#    plt.ylim(0.8, 1)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.title('ROC curve (zoomed in at top left)')
#    plt.legend(loc='best')
#    plt.show()



    # Finding confusion matrix and equal error rate
    y = []
    for i in y_pred_N:
        if i > 0.435:
            i = 1
        else:
            i = 0
        y.append(i)
        
    #print(y)    
    
    cm = confusion_matrix(y_test, y)
    tn, fp, fn, tp = confusion_matrix(y_test, y).ravel()
    print('\nConfusion Matrix :\n', cm)
    print('True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d' %(tn, fp, fn, tp))
    

    
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print("Equal error rate = ", eer)
    
    
## For decision fusion using Geometric Mean

#    y_pred = np.multiply(y_pred_M, y_pred_S)
#    np.multiply(y_pred, y_pred_N, y_pred)
#    np.power(y_pred, 1/3, y_pred)
#    print(y_pred)
#    
#    from sklearn.metrics import roc_curve, auc
#    from sklearn.metrics import roc_auc_score
#    from sklearn.metrics import confusion_matrix
#    from scipy.signal import medfilt
#    y_test = spio.loadmat('Path to .mat file')
#    y_test = y_test["test_label"]
#    y_test = np.array(y_test.reshape(15282))
#    
#    print(y_pred)
#    y_pred = medfilt(y_pred, 11)
#    print(y_pred)
#    
#    print('True labels:', y_test)
#    print('Predicted labels:',y_pred.ravel())
#    auc_score = roc_auc_score(y_test, y_pred)
#    print('AUC score: ', auc_score)
#    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#    AUC = auc(fpr, tpr)
#    
#    y_N = []
#    for i in y_pred_N:
#        if i > 0.435:
#            i = 1
#        else:
#            i = 0
#        y_N.append(i)
#            
#    
#    cm = confusion_matrix(y_test, y_N)
#    tn, fp, fn, tp = confusion_matrix(y_test, y_N).ravel()
#    print('\nConfusion Matrix :\n', cm)
#    print('True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d' %(tn, fp, fn, tp))
#    
#    
#    plt.figure(1)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.title('ROC curve-Decision fusion')
#    plt.legend(loc='best')
#    plt.show()
#
#    
#    from scipy.optimize import brentq
#    from scipy.interpolate import interp1d
#    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#    thresh = interp1d(fpr, thresholds)(eer)
#    print(eer, thresh)
    
    ## MODEL - IV: Feature Fusion (MFCC, Sadjadi and New)    

    batch_size = 128
    data_dim = 60
    timesteps = 547978
    lrate = [0.001]
    
    print("Building network...")
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(Dense(64, input_shape=(None, data_dim)))
    model.add((CuDNNLSTM(128, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add((Dense(1, activation='linear')))
    
    print("Learning rate = ", 0.001)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    model.summary()
    
    print("Loading data...")
    x_train = spio.loadmat('Path to .mat file')
    x_test = spio.loadmat('Path to .mat file')
    y_train = spio.loadmat('Path to .mat file')
    x_val = spio.loadmat('Path to .mat file')
    y_test = spio.loadmat('Path to .mat file')
    y_val = spio.loadmat('Path to .mat file')
    x_train = x_train["train_feat"]
    y_train = y_train["train_label"]
    x_train = x_train.reshape((1, 547978, 60))
    y_train = y_train.reshape((1, 547978, 1))
    x_test = x_test["testing_feat"]
    y_test = y_test["test_label"]
    x_test = x_test.reshape((1, 15282, 60))
    y_test = y_test.reshape((1, 15282, 1))
    x_val = x_val["train_feat"]
    y_val = y_val["train_label"]
    x_val = x_val.reshape((1, 22027, 60))
    y_val = y_val.reshape((1, 22027, 1))
    
    np.save('x_train_All_500.npy', x_train)
    np.save('x_test_All.npy', x_test)
    np.save('y_train_500.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('x_val_All', x_val)
    np.save('y_val', y_val)
    
    x_train = np.load('Path to .npy file')
    x_test = np.load('Path to .npy file')
    y_train = np.load('Path to .npy file')
    y_test = np.load('Path to .npy file')
    x_val = np.load('Path to .npy file')
    y_val = np.load('Path to .npy file')

    print("Training model...")        
    history = model.fit(x_train, y_train, batch_size, epochs=100, validation_data=(x_val, y_val) ,shuffle=True)
    
    model.save('Model_All.h5')
    
    print("Testing model...")
    score = model.evaluate(x_test, y_test, batch_size)
    print('\nFinal Score: [Loss, Accuracy] =', score)
    
    y_pred = model.predict(x_test).ravel()
    print("\nPredicted labels: ",y_pred)
    
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()

    
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from scipy.signal import medfilt
    y_test = spio.loadmat('Path to .mat file')
    y_test = y_test["test_label"]
    y_test = np.array(y_test.reshape(15282))
    
    print(y_pred)
    y_pred = medfilt(y_pred, 11)
    print(y_pred)
    
    print('True labels:', y_test)
    print('Predicted labels:',y_pred.ravel())
    auc_score = roc_auc_score(y_test, y_pred)
    print('AUC score: ', auc_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    AUC = auc(fpr, tpr)
    
    y = []
    for i in y_pred:
        if i > 0.435:
            i = 1
        else:
            i = 0
        y.append(i)
        
    #print(y)    
    
    cm = confusion_matrix(y_test, y)
    tn, fp, fn, tp = confusion_matrix(y_test, y).ravel()
    print('\nConfusion Matrix :\n', cm)
    print('True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d' %(tn, fp, fn, tp))
    
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve-Feature fusion')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
#    plt.figure(2)
#    plt.xlim(0, 0.2)
#    plt.ylim(0.8, 1)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(AUC))
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.title('ROC curve (zoomed in at top left)')
#    plt.legend(loc='best')
#    plt.show()
#    
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print("Equal error rate = ", eer)