# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:25:49 2020

@author: Den
"""

import numpy as np
import random
from math import sin, cos, pi
import matplotlib.pyplot as plt

def generateWTable(N):

    wTable = np.zeros(shape=(N // 2, N // 2))
    for p in range(N // 2):
        for k in range(N // 2):
            wTable[p][k] = cos(4 * pi / N * p * k) + sin(4 * pi / N * p * k)
    return wTable

def generateNewWTable(N):
    newTable = np.zeros(N)
    for p in range(N):
        newTable[p] = cos(2 * pi / N * p) + sin(2 * pi / N * p)
    return newTable


def calculateFFT(N, x):
    
    wTable = generateWTable(N)
    
    newTable = generateNewWTable(N)
    
    FOdd = np.zeros(N // 2)
    
    FEven = np.zeros(N // 2)
    
    F = np.zeros(N)
    
    # calculate odd and even separately
    for p in range(N // 2):
        for k in range(N // 2):
            FEven[p] += x[2 * k] * wTable[p][k]
            FOdd[p] += x[2 * k + 1] * wTable[p][k]
    
    #put it all toghether
    for p in range(N):
        if p < (N // 2):
            F[p] += FEven[p] + newTable[p] * FOdd[p]
        else:
            F[p] += FEven[p - (N // 2)] - newTable[p] * FOdd[p - (N // 2)]
    
    return F


def generateX(n, time, W):
    xValues = [0] * time

    for i in range(1, n + 1):
        amplitude = random.random()
        phase = random.random()

        for t in range(time):
            xValues[t] += amplitude * sin(W / i * (t + 1) + phase)
    return xValues


def main():
    n = 8
    N = 1024
    W = 1200
    
    xValues = generateX(n, N, W)
    
    fftValues = calculateFFT(N, xValues)
        
    figure, (plotXValues, plotFFTValues) = plt.subplots(2, figsize=(20, 20))
    plotXValues.plot(range(N), xValues, "b")
    plotXValues.title.set_text("Згенерований сигнал X")
    plotFFTValues.plot(range(N), fftValues, "r")
    plotFFTValues.title.set_text("FFT")
    plt.show()

if __name__ == '__main__':
    main()
 




