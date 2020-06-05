#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import random
import time
import glob
import joblib
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from scipy import fftpack as fp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from operator import itemgetter
from sklearn.metrics import roc_auc_score, roc_curve

### Function for generating mel-scale filters ###
def melFilterBank(Fs, fftsize, Mel_scale, Mel_cutf, Mel_channel, Mel_norm):
    
    #Define mel-scale parameter m0 based on "1000mel = 1000Hz"
    m0 = 1000.0 / np.log(1000.0 / Mel_scale + 1.0)
    
    #Resolution of frequency
    df = Fs / fftsize
    
    #Mel-scale filters are periodic triangle-shaped structures
    #Define the lower and higher frequency limit of Mel-scale filers
    Nyq = Fs / 2
    f_low, f_high = Mel_cutf
    if f_low is None:
        f_low = 0
    elif f_low < 0:
        f_low = 0
    if f_high is None:
        f_high = Nyq
    elif f_high > Nyq or f_high <= f_low: 
        f_high = Nyq
    #Convert into mel-scale
    mel_Nyq = m0 * np.log(Nyq / Mel_scale + 1.0)
    mel_low = m0 * np.log(f_low / Mel_scale + 1.0)
    mel_high = m0 * np.log(f_high / Mel_scale + 1.0)
    #Convert into index-scale
    n_Nyq = round(fftsize / 2)
    n_low = round(f_low / df)
    n_high = round(f_high / df)
    
    #Calculate the mel-scale interval between triangle-shaped structures
    #Divided by channel+1 because the termination is not the center of triangle but its right edge
    dmel = (mel_high - mel_low) / (Mel_channel + 1)
    
    #List up the center position of each triangle
    mel_center = mel_low + np.arange(1, Mel_channel + 1) * dmel
    
    #Convert the center position into Hz-scale
    f_center = Mel_scale * (np.exp(mel_center / m0) - 1.0)
    
    #Define the center, start, and end position of triangle as index-scale
    n_center = np.round(f_center / df)
    n_start = np.hstack(([n_low], n_center[0 : Mel_channel - 1]))
    n_stop = np.hstack((n_center[1 : Mel_channel], [n_high]))
    
    #Initial condition is defined as 0 padding matrix
    output = np.zeros((n_Nyq, Mel_channel))
    
    #Repeat every channel
    for c in np.arange(0, Mel_channel):
        
        #Slope of a triangle(growing slope)
        upslope = 1.0 / (n_center[c] - n_start[c])
        
        #Add a linear function passing through (nstart, 0) to output matrix 
        for x in np.arange(n_start[c], n_center[c]):
            #Add to output matrix
            x = int(x)
            output[x, c] = (x - n_start[c]) * upslope
        
        #Slope of a triangle(declining slope)
        dwslope = 1.0 / (n_stop[c] - n_center[c])
        
        #Add a linear function passing through (ncenter, 1) to output matrix 
        for x in np.arange(n_center[c], n_stop[c]):
            #Add to output matrix
            x = int(x)
            output[x, c] = 1.0 - ((x - n_center[c]) * dwslope)
        
        #Normalize area underneath each mel-filter into 1
        #T.Ganchev et.al. "Comparative Evaluation of Various MFCC Implementations on the Speaker Verification Task"
        #[ref] https://pdfs.semanticscholar.org/f4b9/8dbd75c87a86a8bf0d7e09e3ebbb63d14954.pdf
        if Mel_norm == True:
            output[:, c] = output[:, c] * 2 / (n_stop[c] - n_start[c])
    
    #Return Mel-scale filters as list (row=frequency, column=Mel channel)
    return output

### Function for getting speech frames ###
def energy_based_VAD(wavdata, FL, FS):
    
    #Construct the frames
    nframes = 1 + np.int(np.floor((len(wavdata) - FL) / FS))
    frames = np.zeros((nframes, FL))
    for i in range(nframes):
        frames[i] = wavdata[i*FS : i*FS + FL]
    
    #Multiply the Hamming window
    HMW = sg.hamming(FL)
    HMW = HMW[np.newaxis, :]
    HMW = np.tile(HMW, (nframes, 1))
    frames = frames * HMW
    
    #Calculate the wave energy std
    S = 20 * np.log10(np.std(frames, axis=1) + 1e-9)
    maxS = np.amax(S)
    
    #Estimate the indices of speech frames
    VAD_index = np.where((S > maxS-30) & (S > -55))
    VAD_index = np.squeeze(np.array(VAD_index))
    
    return VAD_index

### Function for calculating MFCC ###
def get_MFCC(folder_path, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, MFCC_num, MFCC_lifter, Add_deriv, VAD_drop):
    
    #Inicialize list
    x = []
    
    #Get .wav files as an object
    files = glob.glob(folder_path + "/*.wav")
    print("Folder:" + folder_path)
    
    #For a progress bar
    nfiles = len(files)
    unit = math.floor(nfiles/20)
    bar = "#" + " " * math.floor(nfiles/unit)
    
    #Repeat every file-name
    for i, file in enumerate(files):
        
        #Display a progress bar
        print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        if i % unit == 0:
            bar = "#" * math.ceil(i/unit) + " " * math.floor((nfiles-i)/unit)
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        
        #Read the .wav file
        data, Fs = sf.read(file)
        #Transform stereo into monoral
        if(isinstance(data[0], list) == True):
            wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
        else:
            wavdata = data
        
        #Down sampling and normalization of the wave
        wavdata = sg.resample_poly(wavdata, 8000, Fs)
        Fs = 8000
        wavdata = (wavdata - np.mean(wavdata))
        wavdata = wavdata / np.amax(np.abs(wavdata))
        
        #Calculate the index of window size and overlap
        FL = round(frame_length * Fs)
        FS = round(frame_shift * Fs)
        OL = FL - FS
        
        #Call my function for getting speech frames
        VAD_index = energy_based_VAD(wavdata, FL, FS)
        
        #Pass through a pre-emphasis fileter to emphasize the high frequency
        wavdata = sg.lfilter([1.0, -0.97], 1, wavdata)
        
        #Execute STFT
        F, T, dft = sg.stft(wavdata, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
        Adft = np.abs(dft)[0 : round(FL/2)]**2
        
        #Call my function for generating mel-scale filters(row: fftsize/2, column: Channel)
        filterbank = melFilterBank(Fs, FL, Mel_scale, Mel_cutf, Mel_channel, Mel_norm)
        
        #Multiply the filters into the STFT amplitude, and get logarithm of it
        meldft = Adft.T @ filterbank
        if np.any(meldft == 0):
            meldft = np.where(meldft == 0, 1e-9, meldft)
        meldft = np.log10(meldft)
        
        #Get the DCT coefficients (DCT: Discrete Cosine Transformation)
        dct = fp.realtransforms.dct(meldft, type=2, norm="ortho", axis=-1)
        
        #Trim the MFCC features from C(0) to C(MFCC_num)
        dct = np.array(dct[:, 0:MFCC_num])
        
        #Lift up the cepstrum by sine-function depending on channel
        if MFCC_lifter is not None:
            #K.Paliwal, "Decorrelated and liftered filter-bank energies for robust speech recognition." Eurospeech(1999)
            #[ref] https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
            lifter = 1 + 0.5 * MFCC_lifter * np.sin(np.pi * np.arange(0, MFCC_num) / MFCC_lifter)
            nframes = dct.shape[0]
            lifter = lifter[np.newaxis, :]
            lifter = np.tile(lifter, (nframes, 1)) #Duplicate in order to Hadamard operation
            dct = dct * lifter
        
        #Compute the 1st and 2nd derivatives
        if Add_deriv == True:
            deriv1 = np.diff(dct, n=1, axis=0)[1:] #Adjust the length into 2nd derivative (1:)
            deriv2 = np.diff(dct, n=2, axis=0)
            dct = dct[2:] #Adjust the length into 2nd derivative (2:)
            dct = np.concatenate([dct, deriv1], axis=1)
            dct = np.concatenate([dct, deriv2], axis=1)
        
        #Drop the non-speech frames
        if VAD_drop == True:
            dct = dct[VAD_index, :]
        
        #Add results to lists sequentially
        x.append(dct)
    
    #Finish the progress bar
    bar = "#" * math.ceil(nfiles/unit)
    print("\rProgress:[{0}] {1}/{2} Completed!   ".format(bar, i+1, nfiles), end="")
    print()
    
    #Return the result (as list format not numpy tensor)
    return x

### Function for calculating the gaussian ###
def get_gaussprob(data, w, m, cov):
    
    #data: matrix(nframes x ndim)
    #w: scalar, m: vector(ndim), v: matrix(ndim x ndim)
    
    #Get the number of frames and features
    nframes = data.shape[0]
    ndim = data.shape[1]
    
    #Duplicate mean
    m = m[np.newaxis, :]
    m = np.tile(m, (nframes, 1))
    
    #Avoid the zero-determinant error
    det_cov = abs(np.linalg.det(cov))
    if det_cov < 1e-150:
        cov = cov + 1e-6*np.identity(ndim)
        det_cov = abs(np.linalg.det(cov))
    
    #Calculate the logarithm of gaussian with weight
    C = ndim * np.log(2*np.pi) + np.log(det_cov)
    EXP = (data - m) @ (np.linalg.inv(cov))
    EXP = EXP @ (data - m).T
    EXP = np.diag(EXP)
    gaussprob = -0.5*(EXP + C) + np.log(w)
    
    #Convert into exponential
    gaussprob = np.exp(gaussprob)
    
    #Return the wk*P(x|k) as time-frames vector
    return gaussprob

### Function for Calculating the summation of outer product ###
def sum_outer_product(a, b):
    
    #a: nmix x nframes (matrix)
    #b: nframes x MFCCs (matrix)
    
    #Calculate the Hadamard product b*c
    S = []
    for t in range(b.shape[0]):
        B = b[t, :]
        B = B[:, np.newaxis]
        B2 = B @ B.T #Calculate outer product
        S.append(B2)
    S = np.array(S) #S: nframes x MFCCs x MFCCs (tensor)
    
    #Calculate the tensor multiplication
    S = np.transpose(S, (1,0,2)) #S: MFCCs x nframes x MFCCs (tensor)
    S = a @ S #S: MFCCs x nmix x MFCCs (tensor)
    S = np.transpose(S, (1,0,2)) #S: nmix x MFCCs x MFCCs (tensor)
    return S

### Function for MAP (Maximum A Posteriori) adaption ###
def MAP_adaption(train_x, nmix, r, w_UBM, m_UBM, cov_UBM):
    
    #Check the parameter
    if train_x[0].ndim != 2:
        print("Each data should be 2-dimension tensor(time,feature).")
        sys.exit()
    
    #Get the number of audio files and MFCCs
    nfiles = len(train_x)
    ndim = train_x[0].shape[1]
    
    #Get the start time
    start = time.time()
    
    #Initialize variable
    total_frames = 0
    N = np.zeros(nmix)
    F = np.zeros((nmix, ndim))
    S = np.zeros((nmix, ndim, ndim))
    
    #Repeat for each audio file
    for d in range(nfiles):
        
        #Get one data (nframes x MFCCs matrix)
        data = train_x[d]
        
        #Total frames is summation of nframe in dataset
        nframes = data.shape[0] #nframes is diffrent from audio to audio
        total_frames = total_frames + nframes
        
        #Initialize the variable
        G = np.zeros((nmix, nframes))
        L = np.zeros(nframes)
        
        #Initialize responsibility(γ)
        gamma = np.zeros((nmix, nframes))
        
        #Repeat every Gaussian
        for mix in range(nmix):
            
            #Call my function for calculating the Gaussian
            G[mix] = get_gaussprob(data, w_UBM[mix], m_UBM[mix, :], cov_UBM[mix, :, :])
            
            #Get the summation of Gaussian (Likelihood for one data)
            L = L + G[mix]
        
        #Avoid the zero division error
        if np.any(L == 0):
            #print("MAP data No." + str(d) + ": add 1e-6 to likelihood to avoid zero division error.")
            L = np.where(L == 0, 1e-6, L)
        
        #Calcurate the responsibility(γ)
        L = L[np.newaxis, :]
        L = np.tile(L, (nmix, 1)) #Duplicate in order to Hadamard operation
        gamma = G / L
        
        ### Expectation Step (Summation along with time) ###
        #data: nframes x MFCCs (matrix), gamma: nmix x nframes (matrix)
        N = N + np.sum(gamma, axis=1) #Get the summation along with time axis
        F = F + gamma @ data #Calculate the multiplication of gamma by data
        S = S + sum_outer_product(gamma, data) #Calculate the summation of outer product
    
    ### Maximization Step (MAP adaption) ###
    #Compute the alpha
    alpha = N / (N + r)  #alpha: nmix (vector)
    
    #Adapt the weight
    w_ML = N / total_frames #w_ML: nmix (vector)
    w = alpha * w_ML + (1 - alpha) * w_UBM #w: nmix (vector)
    w = w / np.sum(w) #Normalization
    
    #Avoid the zero division error
    if np.any(N == 0):
        #print("Add 1e-6 to N to avoid zero division error.")
        N = np.where(N == 0, 1e-6, N)
    
    #Adapt the mean
    N2 = N[:, np.newaxis]
    N2 = np.tile(N2, (1, ndim)) #Duplicate in order to Hadamard operation
    m_ML = F / N2 #m_ML: nmix x MFCCs (matrix)
    alpha2 = alpha[:, np.newaxis]
    alpha2 = np.tile(alpha2, (1, ndim)) #Duplicate in order to Hadamard operation
    m = alpha2 * m_ML + (1 - alpha2) * m_UBM #m: nmix x MFCCs (matrix)
    
    #Adapt the covariance
    N3 = N2[:, :, np.newaxis]
    N3 = np.tile(N3, (1, 1, ndim)) #Duplicate in order to Hadamard operation
    cov_ML = S / N3 #cov_ML: nmix x MFCCs x MFCCs (tensor)
    alpha3 = alpha2[:, :, np.newaxis]
    alpha3 = np.tile(alpha3, (1, 1, ndim)) #Duplicate in order to Hadamard operation
    Imix = np.identity(nmix) #The function "sum_outer_product" results in outer product by using Imix
    cov = alpha3 * cov_ML + (1 - alpha3) * (cov_UBM + sum_outer_product(Imix, m_UBM)) - sum_outer_product(Imix, m)
    
    #Return the Gaussian parameter after MAP adaption
    return w, m, cov

### Function for calculating the score from UBM and MAP-adaption ###
def get_score(test_x, nmix, w, m, cov, w_UBM, m_UBM, cov_UBM):
    
    #Check the parameter
    if test_x[0].ndim != 2:
        print("Each data should be 2-dimension tensor(time,feature).")
        sys.exit()
    
    #Get the number of audio files and MFCCs
    nfiles = len(test_x)
    ndim = test_x[0].shape[1]
    
    #Initialize the score (nfiles vector)
    score = np.zeros(nfiles)
    
    #Repeat every audio file
    for d in range(nfiles):
        
        #Get one data (nframes x MFCCs matrix)
        data = test_x[d]
        
        #Total frames is summation of nframe in all dataset
        nframes = data.shape[0] #nframes is diffrent from audio to audio
        
        #Initialize the variable
        G_UBM = np.zeros((nmix, nframes))
        G_MAP = np.zeros((nmix, nframes))
        L_UBM = np.zeros(nframes)
        L_MAP = np.zeros(nframes)
        
        #Repeat every Gaussian
        for mix in range(nmix):
            
            #Call my function for calculating the Gaussian
            G_UBM[mix] = get_gaussprob(data, w_UBM[mix], m_UBM[mix, :], cov_UBM[mix, :, :])
            G_MAP[mix] = get_gaussprob(data, w[mix], m[mix, :], cov[mix, :, :])
            
            #Get the summation of Gaussian (Likelihood for one data)
            L_UBM = L_UBM + G_UBM[mix]
            L_MAP = L_MAP + G_MAP[mix]
        
        #Avoid log(0) error
        if np.any(L_MAP == 0):
            #print("Test data No." + str(d) + ": add 1e-6 to likelihood to avoid log(0) error.")
            L_MAP = np.where(L_MAP == 0, 1e-6, L_MAP)
        
        #Avoid log(0) error
        if np.any(L_UBM == 0):
            #print("Test data No." + str(d) + ": add 1e-6 to likelihood to avoid log(0) error.")
            L_UBM = np.where(L_UBM == 0, 1e-6, L_UBM)
        
        #Calculate the score (Time-averaged log-scale likelihood)
        score[d] = np.sum(np.log(L_MAP) - np.log(L_UBM)) / nframes
    
    #Return the score as nfiles vector
    return score

### Function for calculating AUC(Area Under ROC Curve) and its standard error ###
def get_AUC(test_y, score, true_cl, fold):
    
    #Compute the AUC
    AUC = roc_auc_score(test_y, score)
    
    #Plot the ROC curve
    #plt.rcParams["font.size"] = 16
    #plt.figure(figsize=(12, 8))
    #fpr, tpr, thresholds = roc_curve(test_y, score)
    #plt.plot([0, 1], [0, 1], linestyle='--')
    #plt.plot(fpr, tpr, marker='.')
    #plt.title('ROC curve')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.savefig("./log/" + true_cl + "_" + str(fold+1) + "ROCcurve.png")
    #plt.show()
    
    #Return AUC
    return AUC

### Main ###
if __name__ == "__main__":
    
    #Set up
    frame_length = 0.032   #STFT window width(second) [Default]0.032
    frame_shift = 0.016    #STFT window shift(second) [Default]0.016
    Mel_scale = 700        #Mel-frequency is proportional to "log(f/Mel_scale + 1)" [Default]700
    Mel_cutf = [0, None]   #The cutoff frequency of mel-filter [Default] [0, None(Nyquist)]
    Mel_channel = 27       #The number of frequency channel for mel-scale filters [Default]27
    Mel_norm = False       #Normalize the area underneath each mel-filter into 1 [Default]False
    MFCC_num = 13          #The number of MFCCs including C(0) [Default]13
    MFCC_lifter = None     #MFCCs are lifted by "1+0.5*lifter*sin(pie*DCT_order/lifter)" [Default]None or 22
    Add_deriv = True       #Add 1st and 2nd derivatives of MFCCs [Default]True
    VAD_drop = True        #Drop non-speech frames by voice activity detection [Default]True
    r = 9                  #Relevant parameter for MAP adaption [Default]9
    cv = 10                #The number of folds for cross varidation [Default]10
    mode = 0               #0: calculate MFCCs from the beginning, 1: read local files [Default]0
    
    #The number of Gaussian mixture
    #mix_list = [4] #for a quick test
    mix_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    #Get folder names to classify
    folder_path = "./audio_data/MAP_adaption"
    classes = []
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)): #names of folder only
            classes.append(folder)
    
    #In case of calculating the MFCCs from audio
    x_classes = []
    if mode == 0:
        #Repeat every classes
        for cl in classes:
            #Calculating MFCCs of true-class data
            fpath = folder_path + "/" + cl
            x = get_MFCC(fpath, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, MFCC_num, MFCC_lifter, Add_deriv, VAD_drop)
            
            #Preserve the MFCCs as list (not numpy tensor)
            fpath = "./joblib_files/" + cl + "_MFCCs.txt"
            f = open(fpath, 'wb')
            joblib.dump(x, f, compress=3)
            f.close()
            x_classes.append(x)
    
    #In case of reading list from local file 
    else:
        #Repeat every classes
        for cl in classes:
            #Read the data from local file
            folder_path = "./joblib_files/" + cl + '_MFCCs.txt'
            f = open(folder_path, 'rb')
            x = joblib.load(f)
            f.close()
            x_classes.append(x)
    
    #Define parameters for cross validation
    total_files = len(x_classes[0])
    test_files = math.floor(total_files / cv)
    enu = list(range(total_files))
    
    #Initialize the tensor for AUC score
    AUC_tensor = np.zeros((len(mix_list), 4, cv))
    
    #Repeat every fold
    for fold in range(cv):
        
        #Get randomly test sampling without replacement
        test_i = random.sample(enu, k=test_files)
        train_i = list(set(enu) - set(test_i)) #The remain is for training
        
        #Repeat every class
        for true_cl in range(len(classes)):
            
            #Get the test data and class label for true class
            true_x = x_classes[true_cl]
            test_x = list(itemgetter(*test_i)(true_x))
            test_y = np.ones(test_files, dtype=np.int) #true class=1
            train_x = list(itemgetter(*train_i)(true_x))
            
            #Extend the test data and class label for other class
            for cl in range(len(classes)):
                
                #Except for true-class
                if cl != true_cl:
                    
                    #Extend the test data
                    x = x_classes[cl]
                    x = list(itemgetter(*test_i)(x))
                    test_x.extend(x)
                    
                    #Extend the class label
                    y = np.zeros(test_files, dtype=np.int) #other class=0
                    test_y = np.concatenate([test_y, y], axis=0)
            
            #Repeat every mixture number
            for i, nmix in enumerate(mix_list):
                
                #Get the start time
                start = time.time()
                
                #Read the pre-learned UBM model
                folder_path = "./models"
                w_UBM = np.load(folder_path + '/ubm_' + str(nmix) + 'mixGauss_weight.npy')
                m_UBM = np.load(folder_path + '/ubm_' + str(nmix) + 'mixGauss_mean.npy')
                cov_UBM = np.load(folder_path + '/ubm_' + str(nmix) + 'mixGauss_covariance.npy')
                
                #Call my function for MAP adaption
                [w, m, cov] = MAP_adaption(train_x, nmix, r, w_UBM, m_UBM, cov_UBM)
                
                #Call my function for calculating the score (vector)
                score = get_score(test_x, nmix, w, m, cov, w_UBM, m_UBM, cov_UBM)
                
                #Call my function for calculating the AUC
                A = get_AUC(test_y, score, true_cl, fold)
                AUC_tensor[i, true_cl, fold] = A
                
                #Output the result of each fold
                finish = time.time() - start
                print('Fold{}: N_mix={}, AUC_{}={:.5f}, Process_time={:.1f}sec'.format(fold+1, nmix, classes[true_cl], A, finish))
                
                #Define the folder path
                log_path = "./log/" + str(nmix) + "mixGauss.txt"
                if os.path.isfile(log_path):
                    with open(log_path, "a") as f:
                        f.write('{}\t{}\t{:.5f}\n'.format(classes[true_cl], fold+1, A))
                else:
                    with open(log_path, "w") as f:
                        f.write('{}\t{}\t{:.5f}\n'.format(classes[true_cl], fold+1, A))
    
    #Average the result of cv-folds
    AUC = np.average(AUC_tensor, axis=2)
    SE = np.std(AUC_tensor, axis=2) / np.sqrt(cv-1) #Population variance in cross varidation
    
    #Output the result
    for i, nmix in enumerate(mix_list):
        
        for true_cl in range(4):
            print('N_mix={}, AUC_{}={:.5f}, CI(95%)=±{:.5f}'.format(nmix, classes[true_cl], AUC[i, true_cl], 1.96*SE[i, true_cl]))
            with open(log_path, "a") as f:
                f.write('{}\t{:.5f}\t{:.5f}\n'.format(classes[true_cl], AUC[i, true_cl], 1.96*SE[i, true_cl]))
    
    #Plot the result
    x = list(range(len(mix_list)))
    y = np.arange(0.5, 1, 0.05)
    plt.rcParams["font.size"] = 16
    fig = plt.figure(figsize=(15, 10))
    p1 = plt.errorbar(x, AUC[:, 0], yerr=1.96*SE[:, 0], capsize=5, fmt='o-', markersize=8, ecolor='k', markeredgecolor = "k", color='k')
    p2 = plt.errorbar(x, AUC[:, 1], yerr=1.96*SE[:, 1], capsize=5, fmt='o-', markersize=8, ecolor='g', markeredgecolor = "g", color='g')
    p3 = plt.errorbar(x, AUC[:, 2], yerr=1.96*SE[:, 2], capsize=5, fmt='o-', markersize=8, ecolor='r', markeredgecolor = "r", color='r')
    p4 = plt.errorbar(x, AUC[:, 3], yerr=1.96*SE[:, 3], capsize=5, fmt='o-', markersize=8, ecolor='b', markeredgecolor = "b", color='b')
    plt.legend([p1, p2, p3, p4], ["No contamination", "Distortion", "Noise", "Reverberation"], loc=4, title="The type of degradation")
    plt.xticks(x, mix_list)
    plt.yticks(y)
    plt.title('GMM dependence of AUC score')
    plt.xlabel('The number of Gaussian mixture')
    plt.ylabel('AUC score')
    plt.show()