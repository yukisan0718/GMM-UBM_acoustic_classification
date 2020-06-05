#!/usr/bin/env python
# coding: utf-8

import sys
import math
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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

### Function for generating Mel-scale filters ###
def melFilterBank(Fs, fftsize, Mel_scale, Mel_cutf, Mel_channel, Mel_norm):
    
    #Define Mel-scale parameter m0 based on "1000Mel = 1000Hz"
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
    #Convert into Mel-scale
    mel_Nyq = m0 * np.log(Nyq / Mel_scale + 1.0)
    mel_low = m0 * np.log(f_low / Mel_scale + 1.0)
    mel_high = m0 * np.log(f_high / Mel_scale + 1.0)
    #Convert into index-scale
    n_Nyq = round(fftsize / 2)
    n_low = round(f_low / df)
    n_high = round(f_high / df)
    
    #Calculate the Mel-scale interval between triangle-shaped structures
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
        
        #Normalize area underneath each Mel-filter into 1
        #[ref] T.Ganchev, N.Fakotakis, and G.Kokkinakis, Proc. of SPECOM 1, 191-194 (2005)
        #https://pdfs.semanticscholar.org/f4b9/8dbd75c87a86a8bf0d7e09e3ebbb63d14954.pdf
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
        
        #Call my function for generating Mel-scale filters(row: fftsize/2, column: Channel)
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
    
    #Avoid the underflow of determinant 
    det_cov = abs(np.linalg.det(cov))
    uf = False
    if det_cov < 1e-150:
        cov = cov + 1e-6*np.identity(ndim)
        det_cov = abs(np.linalg.det(cov))
        uf = True
    
    #Calculate the logarithm of gaussian with weight
    C = ndim * np.log(2*np.pi) + np.log(det_cov)
    EXP = (data - m) @ (np.linalg.inv(cov))
    EXP = EXP @ (data - m).T
    EXP = np.diag(EXP)
    gaussprob = -0.5*(EXP + C) + np.log(w)
    
    #Convert into exponential
    gaussprob = np.exp(gaussprob)
    
    #Return the wk*P(x|k) as time-frames vector
    return gaussprob, uf

### Function for Calculating the summation of outer product ###
def sum_outer_product(a, b):
    
    #a: matrix(nmix x nframes)
    #b: matrix(nframes x ndim)
    
    #Calculate the outer product b*b
    S = []
    for t in range(b.shape[0]):
        B = b[t, :]
        B = B[:, np.newaxis]
        B2 = B @ B.T #Calculate outer product
        S.append(B2)
    S = np.array(S) #S: tensor(nframes x ndim x ndim)
    
    #Calculate the tensor multiplication
    S = np.transpose(S, (1,0,2)) #S: tensor(ndim x nframes x ndim)
    S = a @ S #S: tensor(ndim x nmix x ndim)
    S = np.transpose(S, (1,0,2)) #S: tensor(nmix x ndim x ndim)
    return S

### Function for initializing GMM parameters ###
def ini_param(train_x, nmix, ini_mode):
    
    #Get the number of audio files and ndims
    nfiles = len(train_x)
    ndim = train_x[0].shape[1]
    
    #Initialize weight by random number
    w = np.random.rand(nmix) #w: vector(nmix)
    w = w / np.sum(w)
    
    #Initialize mean by global average
    total_frames = 0
    gm = np.zeros(ndim)
    for d in range(nfiles):
        nframes = train_x[d].shape[0]
        total_frames = total_frames + nframes
        gm = gm + np.sum(train_x[d], axis=0)
    gm = gm / total_frames #gm: vector(ndim)
    m = gm[np.newaxis, :]
    m = np.tile(m, (nmix, 1)) #m: matrix(nmix x ndim)
    
    #Initialize covariance by global average
    cov = np.zeros((nmix, ndim, ndim))
    for d in range(nfiles):
        nframes = train_x[d].shape[0]
        gm2 = gm[np.newaxis, :]
        gm2 = np.tile(gm2, (nframes, 1)) #gm2: matrix(nframes x ndim)
        Ones = np.ones((nmix, nframes)) #Use "Ones" to get uniform summation of (x-m)^2
        cov = cov + sum_outer_product(Ones, train_x[d] - gm2) #cov: tensor(nmix x ndim x ndim)
    cov = cov / total_frames
    
    #In case of "kmeans" initialization
    if ini_mode[0:6] == "kmeans":
        #Average with time-frames
        x = np.zeros((nfiles, ndim))
        for d in range(nfiles):
            x[d, :] = np.average(train_x[d], axis=0)
        
        #Initialize mean by k-means++ or normal k-means
        if ini_mode == "kmeans++":
            clf = KMeans(n_clusters=nmix, init='k-means++', n_jobs=4)
        else:
            clf = KMeans(n_clusters=nmix, init='random', n_jobs=4)
        m = clf.fit(x).cluster_centers_
    
    #In case of "random" initialization
    elif ini_mode == "random":
        #Calculate the sigma
        sigma = np.sqrt(np.diag(cov[0])) #sigma: vector(ndim)
        sigma = sigma[np.newaxis, :]
        sigma = np.tile(sigma, (nmix, 1)) #sigma: matrix(nmix x ndim)
        
        #Initialize mean by random number
        rand_mat = 2 * np.random.rand(nmix, ndim) - 1 #random value between -1 and +1
        m = m + 0.5 * rand_mat * sigma
    
    #Others
    else:
        print("UBM_ini should be 'kmeans++' or 'kmeans' or 'random'.")
        sys.exit()
    
    return w, m, cov

### Function for training the UBM (Universal Background Model) ###
def UBM_train(train_x, nmix, max_iter, ini_mode, folder_path):
    
    #Check the parameter
    if train_x[0].ndim != 2:
        print("Each data should be 2-dimension tensor(time,feature).")
        sys.exit()
    
    #Get the number of audio files and features
    nfiles = len(train_x)
    ndim = train_x[0].shape[1]
    
    #Define the delta to avoid underflow
    delta = np.zeros((nmix, ndim, ndim))
    
    #In case of "read" the pre-learned parameters
    if ini_mode == "read":
        #Read the UBM parameters from local files
        w = np.load(folder_path + '/UBM/ubm_' + str(nmix) + 'mixGauss_weight.npy')
        m = np.load(folder_path + '/UBM/ubm_' + str(nmix) + 'mixGauss_mean.npy')
        cov = np.load(folder_path + '/UBM/ubm_' + str(nmix) + 'mixGauss_covariance.npy')
    else:
        #Call my function for initializing GMM parameters
        [w, m, cov] = ini_param(train_x, nmix, ini_mode)
    
    #Repeat every iteration
    score = 0
    i = 0
    while i < max_iter:
        
        #Preserve the UBM parameters temporarily 
        np.save(folder_path + '/UBM/ubm_temp' + str(i+1) + '_mixGauss_weight', w)
        np.save(folder_path + '/UBM/ubm_temp' + str(i+1) + '_mixGauss_mean', m)
        np.save(folder_path + '/UBM/ubm_temp' + str(i+1) + '_mixGauss_covariance', cov)
        
        #Get the start time
        start = time.time()
        
        #Initialize variable
        total_frames = 0
        logL = 0
        N = np.zeros(nmix)
        F = np.zeros((nmix, ndim))
        S = np.zeros((nmix, ndim, ndim))
        
        #For a progress bar
        unit = math.floor(nfiles/20)
        bar = "#" + " " * math.floor(nfiles/unit)
        
        #Repeat every audio file
        for d in range(nfiles):
            
            #Display a progress bar
            print("\rIter{0}:[{1}] {2}/{3} Processing...".format(i+1, bar, d+1, nfiles), end="")
            if d % unit == 0:
                bar = "#" * math.ceil(d/unit) + " " * math.floor((nfiles-d)/unit)
                print("\rIter{0}:[{1}] {2}/{3} Processing...".format(i+1, bar, d+1, nfiles), end="")
            
            #Get one data (matrix: nframes x ndim)
            data = train_x[d]
            
            #Total frames is summation of nframe in all dataset
            nframes = data.shape[0] #nframes is diffrent from audio to audio
            total_frames = total_frames + nframes
            
            #Initialize the variable
            G = np.zeros((nmix, nframes))
            L = np.zeros(nframes)
            gamma = np.zeros((nmix, nframes)) #gannma: num_mix x nframes (matrix)
            
            #Repeat every Gaussian
            for mix in range(nmix):
                
                #Call my function for calculating the Gaussian
                G[mix], uf = get_gaussprob(data, w[mix], m[mix, :], cov[mix, :, :])
                
                #In case of underflow-covariance
                if uf == True:
                    cov[mix] = cov[mix] + 1e-6*np.identity(ndim)
                    delta[mix] = 1e-6*np.identity(ndim)
                
                #Get the summation of Gaussian (Likelihood for one data)
                L = L + G[mix]
            
            #Avoid log(0) error
            if np.any(L == 0):
                #print("Data No." + str(d) + ": add 1e-6 to likelihood to avoid log(0) error.")
                L = np.where(L == 0, 1e-6, L)
            
            #Get the summation of log-scale likelihood
            logL = logL + np.sum(np.log(L)) #logL: scalar
            
            #Calcurate the responsibility(γ)
            L = L[np.newaxis, :]
            L = np.tile(L, (nmix, 1))
            gamma = G / L
            
            ### Expectation step ###
            #data: nframes x ndim (matrix), gamma: nmix x nframes (matrix)
            N = N + np.sum(gamma, axis=1) #Get the summation along with time axis
            F = F + gamma @ data #Calculate the multiplication of gamma by data
            S = S + sum_outer_product(gamma, data) #Calculate the summation of outer product
        
        #Finish the progress bar
        bar = "#" * math.ceil(nfiles/unit)
        print("\rIter{0}:[{1}] {2}/{3} Completed!   ".format(i+1, bar, d+1, nfiles), end="")
        print()
        
        #Avoid the zero division error
        if np.any(N == 0):
            #print("Add 1e-6 to N to avoid zero division error.")
            N = np.where(N == 0, 1e-6, N)
        
        ### Maximization Step ###
        w = N / total_frames #w: nmix (vector)
        N2 = N[:, np.newaxis]
        N2 = np.tile(N2, (1, ndim)) #Duplicate in order to Hadamard operation
        m = F / N2 #m: nmix x ndim (matrix)
        N3 = N2[:, :, np.newaxis]
        N3 = np.tile(N3, (1, 1, ndim)) #Duplicate in order to Hadamard operation
        Imix = np.identity(nmix) #The function "sum_outer_product" results in outer product by using Imix
        cov = S / N3 - sum_outer_product(Imix, m) + delta
        
        #Calculate the total likelihood (score)
        Likelihood = logL / total_frames #Get summation and divide by N,T
        if i == 0:
            diff = 0
        else:
            diff = Likelihood - score #diffrence from the former iteration
        score = Likelihood
        
        #Output the result and process time
        finish = time.time() - start
        print("N_mix={}, Iter{}, Likelihood={:.5f}, Gain={:.5f}, Process_time={:.1f}sec".format(nmix, i+1, score, diff, finish))
        
        #Condition of convergence
        if i != 0 and diff < 1e-3:
            break
        i = i + 1
    
    #Return the Gaussian Parameter for UBM
    return w, m, cov

### Main ###
if __name__ == "__main__":
    
    #Set up
    frame_length = 0.032   #STFT window width(second) [Default]0.032
    frame_shift = 0.016    #STFT window shift(second) [Default]0.016
    Mel_scale = 700        #Mel-frequency is proportional to "log(f/Mel_scale + 1)" [Default]700 or 1000
    Mel_cutf = [0, None]   #The cutoff frequency of Mel-filter [Default] [0, None(Nyquist)]
    Mel_channel = 27       #The number of frequency channel for Mel-scale filters [Default]27
    Mel_norm = False       #Normalize the area underneath each Mel-filter into 1 [Default]False
    MFCC_num = 13          #The number of MFCCs including C(0) [Default]13
    MFCC_lifter = None     #MFCCs are lifted by "1+0.5*lifter*sin(pie*DCT_order/lifter)" [Default]None or 22
    Add_deriv = True       #Add 1st and 2nd derivatives of MFCCs [Default]True
    VAD_drop = True        #Drop non-speech frames by voice activity detection [Default]True
    UBM_ini = "kmeans++"   #How to initialize the Gaussian parameters(kmeans++, kmeans, random, read)
    mode = 0               #0: calculate the MFCCs, 1: read local files, -1: generate demo data
    
    #The number of Gaussian mixture and iteration
    #mix_list = [4] #for a quick test
    #UBM_iter = [2] #for a quick test
    mix_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    UBM_iter = [2, 2, 4, 4, 4, 4, 6, 6, 10, 10, 15]
    
    #In case of calculation MFCCs from audio
    if mode == 0:
        #Calculating MFCCs
        folder_path = "./audio_data/UBM_training"
        train_x = get_MFCC(folder_path, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, MFCC_num, MFCC_lifter, Add_deriv, VAD_drop)
        
        #Preserve the MFCCs as list (not numpy tensor)
        folder_path = "./joblib_files"
        folder_path = folder_path + "/ubm_MFCCs.txt"
        f = open(folder_path, 'wb')
        joblib.dump(train_x, f, compress=3)
        f.close()
    
    #In case of reading list from local file
    elif mode == 1:
        #Read the training data from local files
        folder_path = "./joblib_files"
        folder_path = folder_path + "/ubm_MFCCs.txt"
        f = open(folder_path, 'rb')
        train_x = joblib.load(f)
        f.close()
    
    #Test data for verifying GMM code
    else:
        #Generate demo data
        N = 300
        train_x = np.concatenate([np.random.multivariate_normal([-2, 0], np.eye(2), round(N/3)),
                np.random.multivariate_normal([0, 5], np.eye(2), round(N/3)),
                np.random.multivariate_normal([4, 3], np.eye(2), round(N/3))])
        train_x = train_x[:, np.newaxis, :]
        train_x = np.tile(train_x, (1, 100, 1))
    
    #Repeat every mixture number
    for i, nmix in enumerate(mix_list):
        
        #Call my function for training the UBM
        folder_path = "./models"
        [w, m, cov] = UBM_train(train_x, nmix, UBM_iter[i], UBM_ini, folder_path)
        
        #Preserve the UBM parameters as numpy-lists in local files
        np.save(folder_path + '/ubm_' + str(nmix) + 'mixGauss_weight', w)
        np.save(folder_path + '/ubm_' + str(nmix) + 'mixGauss_mean', m)
        np.save(folder_path + '/ubm_' + str(nmix) + 'mixGauss_covariance', cov)
        
        #Plot the Gaussian distribution
        plt.rcParams["font.size"] = 16
        plt.figure(figsize=(12, 8))
        colors = np.repeat(['r', 'g', 'b', 'm'], np.ceil(nmix/4))
        for k in range(nmix):
            plt.scatter(m[k, 0], m[k, 1], c=colors[k], marker='o', zorder=3)
        
        #Plot the contour graph
        nfiles = len(train_x)
        ndim = train_x[0].shape[1]
        ave_x = np.zeros((nfiles, ndim))
        for d in range(nfiles):
            ave_x[d, :] = np.average(train_x[d], axis=0)
        x_min, x_max = np.amin(ave_x[:, 0])-0.5, np.amax(ave_x[:, 0])+0.5
        y_min, y_max = np.amin(ave_x[:, 1])-0.5, np.amax(ave_x[:, 1])+0.5
        xlist = np.linspace(x_min, x_max, 50)
        ylist = np.linspace(y_min, y_max, 50)
        x, y = np.meshgrid(xlist, ylist)
        pos = np.dstack((x,y))
        for k in range(nmix):
            z = multivariate_normal(m[k, 0:2], cov[k, 0:2, 0:2]).pdf(pos)
            cs = plt.contour(x, y, z, 3, colors=colors[k], linewidths=2, zorder=2)
        
        #Plot the time-averaged training data
        plt.plot(ave_x[:, 0], ave_x[:, 1], 'cx', zorder=1)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('GMM distribution by UBM')
        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.show()