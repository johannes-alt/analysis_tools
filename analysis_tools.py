# The first (very long) part defines linear combinations of gaussians (1 up to 60) and a dictionary where the functions are stored with the corresponding number (int) of gaussians as key (e.g. 1: gaus, 2:gaus2, 3:gaus3 ...)
# 
# 
# 
# 
# 
# 


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv

def linear(x,m,c):
    return m*x + c

def gaus(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

def gaus2(x,A1,mu1,sig1,A2,mu2,sig2):
    return gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2)

def gaus3(x,A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3):
    return gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3)

def gaus4(x,A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4):
    return gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4,mu4,sig4)

def gaus5(x,A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5):
    return gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4,mu4,sig4) + gaus(x,A5,mu5,sig5)

def gaus6(x,A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6):
    return gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4,mu4,sig4) + gaus(x,A5,mu5,sig5) + gaus(x,A6,mu6,sig6)

def gaus7(x,
          A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6,
          A7,mu7,sig7):
    
    g = gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4,mu4,sig4) + gaus(x,A5,mu5,sig5) + gaus(x,A6,mu6,sig6) + \
        gaus(x,A7,mu7,sig7)
    
    return g

def gaus8(x,
          A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6,
          A7,mu7,sig7,A8,mu8,sig8):
    
    g = gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4,mu4,sig4) + gaus(x,A5,mu5,sig5) + gaus(x,A6,mu6,sig6) + \
        gaus(x,A7,mu7,sig7) + gaus(x,A8,mu8,sig8)
    
    return g

def gaus9(x,
          A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6,
          A7,mu7,sig7,A8,mu8,sig8,A9,mu9,sig9):
    
    g = gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4,mu4,sig4) + gaus(x,A5,mu5,sig5) + gaus(x,A6,mu6,sig6) + \
        gaus(x,A7,mu7,sig7) + gaus(x,A8,mu8,sig8) + gaus(x,A9,mu9,sig9)
    
    return g

def gaus10(x,
          A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6,
          A7,mu7,sig7,A8,mu8,sig8,A9,mu9,sig9,A10,mu10,sig10):
    
    g = gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5,mu5,sig5) + gaus(x,A6,mu6,sig6) + \
        gaus(x,A7,mu7,sig7) + gaus(x,A8,mu8,sig8) + gaus(x,A9,mu9,sig9) + gaus(x,A10,mu10,sig10)
    
    return g

def gaus11(x,
          A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6,
          A7,mu7,sig7,A8,mu8,sig8,A9,mu9,sig9,A10,mu10,sig10,A11,mu11,sig11):
    
    g = gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6,mu6,sig6) + \
        gaus(x,A7,mu7,sig7) + gaus(x,A8,mu8,sig8) + gaus(x,A9,mu9,sig9) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11)
    
    return g

def gaus12(x,
          A1,mu1,sig1,A2,mu2,sig2,A3,mu3,sig3,A4,mu4,sig4,A5,mu5,sig5,A6,mu6,sig6,
          A7,mu7,sig7,A8,mu8,sig8,A9,mu9,sig9,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12):
    
    g = gaus(x,A1,mu1,sig1) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7,mu7,sig7) + gaus(x,A8,mu8,sig8) + gaus(x,A9,mu9,sig9) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12)
    
    return g

def gaus13(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2,mu2,sig2) + gaus(x,A3,mu3,sig3) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8,mu8,sig8) + gaus(x,A9,mu9,sig9) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13)
    
    return g

def gaus14(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3,mu3,sig3) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9,mu9,sig9) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14)
    
    return g

def gaus15(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15)
    
    return g

def gaus16(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16)
    
    return g

def gaus17(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17)
    
    return g

def gaus18(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18)
    
    return g

def gaus19(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19)
    
    return g

def gaus20(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20)
    
    return g

def gaus21(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21)
    
    return g

def gaus22(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22)
    
    return g

def gaus23(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23)
    
    return g

def gaus24(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24)
    
    return g

def gaus25(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25)
    
    return g

def gaus26(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26)
    
    return g

def gaus27(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27)
    
    return g

def gaus28(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28)
    
    return g

def gaus29(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29)
    
    return g

def gaus30(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30)
    
    return g

def gaus31(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31)
    
    return g

def gaus32(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32)
    
    return g

def gaus33(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33)
    
    return g

def gaus34(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34)
    
    return g

def gaus35(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35)
    
    return g

def gaus36(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36)
    
    return g

def gaus37(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37)
    
    return g

def gaus38(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38)
    
    return g

def gaus39(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39)
    
    return g

def gaus40(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40)
    
    return g

def gaus41(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41)
    
    return g

def gaus42(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42)
    
    return g

def gaus43(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43)
    
    return g

def gaus44(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44)
    
    return g

def gaus45(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45)
    
    return g

def gaus46(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46)
    
    return g

def gaus47(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47)
    
    return g

def gaus48(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48)
    
    return g

def gaus49(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49)
    
    return g

def gaus50(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50)
    
    return g

def gaus51(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51)
    
    return g

def gaus52(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52)
    
    return g

def gaus53(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53)
    
    return g

def gaus54(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54)
    
    return g

def gaus55(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54,
          A55,mu55,sig55):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54) + \
        gaus(x,A55,mu55,sig55)
    
    return g

def gaus56(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54,
          A55,mu55,sig55,A56,mu56,sig56):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54) + \
        gaus(x,A55,mu55,sig55) + gaus(x,A56,mu56,sig56)
    
    return g

def gaus57(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54,
          A55,mu55,sig55,A56,mu56,sig56,A57,mu57,sig57):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54) + \
        gaus(x,A55,mu55,sig55) + gaus(x,A56,mu56,sig56) + gaus(x,A57,mu57,sig57)
    
    return g

def gaus58(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54,
          A55,mu55,sig55,A56,mu56,sig56,A57,mu57,sig57,A58,mu58,sig58):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54) + \
        gaus(x,A55,mu55,sig55) + gaus(x,A56,mu56,sig56) + gaus(x,A57,mu57,sig57) + gaus(x,A58,mu58,sig58)
    
    return g

def gaus59(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54,
          A55,mu55,sig55,A56,mu56,sig56,A57,mu57,sig57,A58,mu58,sig58,A59,mu59,sig59):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54) + \
        gaus(x,A55,mu55,sig55) + gaus(x,A56,mu56,sig56) + gaus(x,A57,mu57,sig57) + gaus(x,A58,mu58,sig58) + gaus(x,A59,mu59,sig59)
    
    return g

def gaus60(x,
          A1 ,mu1 ,sig1 ,A2 ,mu2 ,sig2 ,A3 ,mu3 ,sig3 ,A4 ,mu4 ,sig4 ,A5 ,mu5 ,sig5 ,A6 ,mu6 ,sig6 ,
          A7 ,mu7 ,sig7 ,A8 ,mu8 ,sig8 ,A9 ,mu9 ,sig9 ,A10,mu10,sig10,A11,mu11,sig11,A12,mu12,sig12,
          A13,mu13,sig13,A14,mu14,sig14,A15,mu15,sig15,A16,mu16,sig16,A17,mu17,sig17,A18,mu18,sig18,
          A19,mu19,sig19,A20,mu20,sig20,A21,mu21,sig21,A22,mu22,sig22,A23,mu23,sig23,A24,mu24,sig24,
          A25,mu25,sig25,A26,mu26,sig26,A27,mu27,sig27,A28,mu28,sig28,A29,mu29,sig29,A30,mu30,sig30,
          A31,mu31,sig31,A32,mu32,sig32,A33,mu33,sig33,A34,mu34,sig34,A35,mu35,sig35,A36,mu36,sig36,
          A37,mu37,sig37,A38,mu38,sig38,A39,mu39,sig39,A40,mu40,sig40,A41,mu41,sig41,A42,mu42,sig42,
          A43,mu43,sig43,A44,mu44,sig44,A45,mu45,sig45,A46,mu46,sig46,A47,mu47,sig47,A48,mu48,sig48,
          A49,mu49,sig49,A50,mu50,sig50,A51,mu51,sig51,A52,mu52,sig52,A53,mu53,sig53,A54,mu54,sig54,
          A55,mu55,sig55,A56,mu56,sig56,A57,mu57,sig57,A58,mu58,sig58,A59,mu59,sig59,A60,mu60,sig60):
    
    g = gaus(x,A1 ,mu1 ,sig1 ) + gaus(x,A2 ,mu2 ,sig2 ) + gaus(x,A3 ,mu3 ,sig3 ) + gaus(x,A4 ,mu4 ,sig4 ) + gaus(x,A5 ,mu5 ,sig5 ) + gaus(x,A6 ,mu6 ,sig6 ) + \
        gaus(x,A7 ,mu7 ,sig7 ) + gaus(x,A8 ,mu8 ,sig8 ) + gaus(x,A9 ,mu9 ,sig9 ) + gaus(x,A10,mu10,sig10) + gaus(x,A11,mu11,sig11) + gaus(x,A12,mu12,sig12) + \
        gaus(x,A13,mu13,sig13) + gaus(x,A14,mu14,sig14) + gaus(x,A15,mu15,sig15) + gaus(x,A16,mu16,sig16) + gaus(x,A17,mu17,sig17) + gaus(x,A18,mu18,sig18) + \
        gaus(x,A19,mu19,sig19) + gaus(x,A20,mu20,sig20) + gaus(x,A21,mu21,sig21) + gaus(x,A22,mu22,sig22) + gaus(x,A23,mu23,sig23) + gaus(x,A24,mu24,sig24) + \
        gaus(x,A25,mu25,sig25) + gaus(x,A26,mu26,sig26) + gaus(x,A27,mu27,sig27) + gaus(x,A28,mu28,sig28) + gaus(x,A29,mu29,sig29) + gaus(x,A30,mu30,sig30) + \
        gaus(x,A31,mu31,sig31) + gaus(x,A32,mu32,sig32) + gaus(x,A33,mu33,sig33) + gaus(x,A34,mu34,sig34) + gaus(x,A35,mu35,sig35) + gaus(x,A36,mu36,sig36) + \
        gaus(x,A37,mu37,sig37) + gaus(x,A38,mu38,sig38) + gaus(x,A39,mu39,sig39) + gaus(x,A40,mu40,sig40) + gaus(x,A41,mu41,sig41) + gaus(x,A42,mu42,sig42) + \
        gaus(x,A43,mu43,sig43) + gaus(x,A44,mu44,sig44) + gaus(x,A45,mu45,sig45) + gaus(x,A46,mu46,sig46) + gaus(x,A47,mu47,sig47) + gaus(x,A48,mu48,sig48) + \
        gaus(x,A49,mu49,sig49) + gaus(x,A50,mu50,sig50) + gaus(x,A51,mu51,sig51) + gaus(x,A52,mu52,sig52) + gaus(x,A53,mu53,sig53) + gaus(x,A54,mu54,sig54) + \
        gaus(x,A55,mu55,sig55) + gaus(x,A56,mu56,sig56) + gaus(x,A57,mu57,sig57) + gaus(x,A58,mu58,sig58) + gaus(x,A59,mu59,sig59) + gaus(x,A60,mu60,sig60)
    
    return g



gaus_func = {1 :gaus  ,2 :gaus2 ,3 :gaus3 ,4 :gaus4 ,5 :gaus5 ,6 :gaus6 ,
             7 :gaus7 ,8 :gaus8 ,9 :gaus9 ,10:gaus10,11:gaus11,12:gaus12,
             13:gaus13,14:gaus14,15:gaus15,16:gaus16,17:gaus17,18:gaus18,
             19:gaus19,20:gaus20,21:gaus21,22:gaus22,23:gaus23,24:gaus24,
             25:gaus25,26:gaus26,27:gaus27,28:gaus28,29:gaus29,30:gaus30,
             31:gaus31,32:gaus32,33:gaus33,34:gaus34,35:gaus35,36:gaus36,
             37:gaus37,38:gaus38,39:gaus39,40:gaus40,41:gaus41,42:gaus42,
             43:gaus43,44:gaus44,45:gaus45,46:gaus46,47:gaus47,48:gaus48,
             49:gaus49,59:gaus59,51:gaus51,52:gaus52,53:gaus53,54:gaus54,
             55:gaus55,56:gaus56,57:gaus57,58:gaus58,59:gaus59,60:gaus60
             }

def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]

class analysis():
    
    # A class to load data, plot data, find peaks and fit peaks
    
    
    def __init__(self,file=None):
        self.bins            = np.linspace(0,2**12-1,2**12)-0.5
        self.bins_edges      = np.linspace(0,2**12,2**12+1)-0.5
        self.bin_range       = [900,1300]
        self.x               = np.linspace(0,2**12,100000)
        self.peak_distance   = 25
        self.peak_prominence = 10
        self.peak_height     = 1
        self.peak_width      = 1
        self.fit_range       = [0,-1] # define the range in which you want to fit
        self.noise_peak      = None # noise peak position
        if file:
            self.file        = file
        else:
            print('No file selected')
    
    def load_data(self):
        try:
            self.data = np.genfromtxt(self.file)[1:,2]
        except:
            data = []
            with open(self.file) as f:
                lines = csv.reader((line.replace('\0','') for line in f), delimiter=' ')
                for row in lines:
                    try:
                        data.append(row[2])
                    except:
                        pass
            self.data = np.array([int(i) for i in data[1:]])
            
    def create_hist(self):
        # create the histogram from data
        # maybe plot??? idk
        self.hist, self.bin_edges = np.histogram(self.data,bins=self.bins_edges)
        self.bin_center = [(self.bin_edges[i]+self.bin_edges[i+1])/2 for i in range(len(self.bin_edges)-1)]
        
    def find_peaks(self):
        # find the peaks in the histogram
        self.peaks, self.peak_prop = find_peaks(self.hist,distance=self.peak_distance,prominence=self.peak_prominence,height=self.peak_height,width=self.peak_width)
        self.n_peaks               = len(self.peaks)
        
    def fit_peaks(self):
        # create p0 and bounds and fit peaks based on the find_peaks results
        
        p0           = [[self.peak_prop['peak_heights'][i]/2,self.peaks[i],10] for i in range(len(self.peaks))]
        
        bounds_lower = [[0,self.peaks[i]-10,0] for i in range(len(self.peaks))]
        bounds_upper = [[np.inf,self.peaks[i]+10,20] for i in range(len(self.peaks))]
        
        bounds_lower = [i for j in bounds_lower for i in j]
        bounds_upper = [i for j in bounds_upper for i in j]
        
        self.p0 = [i for j in p0 for i in j]
        self.bounds = [bounds_lower,bounds_upper]
            
        self.popt, self.pcov = curve_fit(gaus_func[self.n_peaks],self.bin_center[self.fit_range[0]:self.fit_range[1]],self.hist[self.fit_range[0]:self.fit_range[1]],
                                         p0=self.p0,bounds=self.bounds,maxfev=100000)
        
    def analyse(self,load = False):
        if load:
            load_data()
        create_hist()
        find_peaks()
        fit_peaks()
        
        
    def find_first_photon_number(self,cut_last=0):
        if not self.noise_peak:
            self.noise_peak = 962
            print(f'Noise peak position assumed to be {self.noise_peak}')
        
        if cut_last == 0:
            self.gain_popt, self.gain_pcov = curve_fit(linear,np.linspace(0,self.n_peaks-1,self.n_peaks),np.reshape(self.popt,(self.n_peaks,3))[:,1],maxfev=100000)
        
        else:
            self.gain_popt, self.gain_pcov = curve_fit(linear,np.linspace(0,self.n_peaks-1-cut_last,self.n_peaks-cut_last),np.reshape(self.popt,(self.n_peaks,3))[:-cut_last,1],maxfev=100000)
        
        self.first_photon_number = round((self.gain_popt[1]-self.noise_peak)/self.gain_popt[0])
        
        self.gain                = self.gain_popt[0]
        self.noise_pos_from_fit  = linear(-self.first_photon_number,*self.gain_popt)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     