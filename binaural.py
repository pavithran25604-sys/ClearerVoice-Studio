import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import cv2
import os
from pysofaconventions import *
import pyroomacoustics as pra

def stft(xx, win=None, Nw=1024, Nl=512, Nfft=1024, pad_front=True):
    if len(xx.shape) == 1:
        xx = xx[:, None]
    if win is None or win.shape[0] != Nw:
        win = np.sin(np.arange(Nw)/Nw*np.pi)[:, None]
    if len(win.shape) == 1:
        win = win[:, None]
    Npad_front = Nw - Nl if pad_front else 0
    Npad_end = Nw - Nl
    Nx = int(np.ceil(xx.shape[0]/Nl)*Nl) + Npad_front + Npad_end
    x = np.zeros([Nx, xx.shape[1]])
    x[Npad_front:Npad_front+xx.shape[0], :] = xx
    Nb = int((Nx-Nw)/Nl)
    X = np.zeros([Nfft//2+1, xx.shape[1], Nb], dtype=complex)
    i = 0
    b = 0
    while i + Nw < Nx:
        X[:, :, b] = np.fft.rfft(x[i:i+Nw, :]*win, Nfft, axis=0)
        i += Nl
        b += 1
    return X

def istft(X, win=None, Nw=1024, Nl=512, Nfft=1024):
    if win is None or win.shape[0] != Nw:
        win = np.sin(np.arange(Nw)/Nw*np.pi)[:, None]
    if len(win.shape) == 1:
        win = win[:, None]
    Nk, L, Nb = X.shape
    Nx = Nb*Nl + Nw
    x = np.zeros([Nx, L])
    i = 0
    for b in range(Nb):
        x[i:i+Nw, :] += np.fft.irfft(X[:, :, b], axis=0)[:Nw, :]*win
        i += Nl
    return x

# Parameter setup
c = 340 # speed of light
winsize = 512
hopsize = 512//4
fftsize = 512
winA = np.hamming(winsize)
winS = pra.transform.stft.compute_synthesis_window(winA, hopsize)

# Load source
x, fs = sf.read('chinese_48k.wav')
# Compute stft
X = stft(x, win=winA, Nw=winsize, Nl=hopsize, Nfft=fftsize)

# Head
headSize = 0.14
earPos = np.array([
    [-headSize/2, 0, 0],
    [headSize/2, 0, 0]
])

# Design trajectories
initPos = -90/180*np.pi
r = 0.5 
speed = 0.1*np.pi/fs*hopsize # move 0.1pi in azimuth every second
thetas = initPos + np.arange(X.shape[2])*speed
thetas = np.mod(thetas/np.pi*180, 360)

# Load Sofa
path = 'subject_003.sofa'
sofa = SOFAFile(path,'r')
fs_sofa = sofa.getSamplingRate()
pos_sofa_polar = sofa.getVariableValue('SourcePosition') # note that 0deg is at front, 0-180 indicates left, 180-360 indicates right
hrtf = sofa.getDataIR() # num_of_hrtf, 2, length
hrtf = np.swapaxes(hrtf, 0, 2) # length, 2, num_of_hrtf
hrtf = signal.resample_poly(hrtf, fs, fs_sofa, axis=0)
HRTF = np.fft.rfft(hrtf, fftsize, axis=0)

# Generate binaural signals
Y = np.zeros([X.shape[0], 2, X.shape[2]], dtype=complex)
for i in range(X.shape[2]):
    # Compute idx of hrtf for every frame
    idx = np.argmin((pos_sofa_polar[:,0]-thetas[i])**2)
    Y[..., [i]] = HRTF[..., idx][..., None] @ X[..., [i]]
y = istft(Y, win=winS, Nw=winsize, Nl=hopsize, Nfft=fftsize)

sf.write('output_hrtf.wav', y, fs)