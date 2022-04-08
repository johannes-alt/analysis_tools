import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
from gaussian_functions import *


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
        self.xlim            = [900,1200]
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
            
        self.popt, self.pcov = curve_fit(gauss,self.bin_center[self.fit_range[0]:self.fit_range[1]],self.hist[self.fit_range[0]:self.fit_range[1]],
                                         p0=self.p0,bounds=self.bounds,maxfev=100000)#_func[self.n_peaks]
        
    def analyse(self,load = False):
        if load:
            self.load_data()
        self.create_hist()
        self.find_peaks()
        self.fit_peaks()
        
        
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
        
    
    def plot_data(self):
        fig, ax = plt.subplots(1,2)
        for i in ax:
            i.hist(self.data,bins=self.bins)
            i.plot(self.x,gauss(self.x,*self.popt))
            i.set_xlim(self.xlim)
            i.set_ylim([1,np.max(self.hist)*1.1])
        ax[1].set_yscale('log')
        fig.show()
    
    def save(self,filename=False):
        if filename == False:
            filename = self.directory
        with open(filename+'array_save','wb') as f:
            pickle.dump(self,f)
    
    @staticmethod
    def load(filename):
        with open(filename,'rb') as f:
            return pickle.load(f)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     