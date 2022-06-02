import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
import pickle
import inspect, re


def linear(x,m,c):
    return m*x + c


def gauss(x,*params):
    
    g = 0
    
    for i in range(0,len(params),3):
        g += params[i] * np.exp(-(x-params[i+1])**2/(2*params[i+2]**2))
        
    return g

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
    
    
    def __init__(self,file=None,channel=-1):
        
        ####### define bins, bin edges for hist and x linspace for plotting of fits
        self.bins            = np.linspace(0,2**12-1,2**12)-0.5
        self.bins_edges      = np.linspace(0,2**12,2**12+1)-0.5
        self.bins_center     = np.linspace(0,2**12-1,2**12)
        self.bin_range       = [900,1300]
        self.x               = np.linspace(0,2**12,100000)
        
        ####### define peak finding limitations
        self.peak_distance   = 20
        self.peak_prominence = 10
        self.peak_height     = 1
        self.peak_width      = 2
        
        #######
        self.fit_range       = [0,-1] # define the range in which you want to fit
        self.noise_peak      = None # noise peak position
        self.xlim            = [900,1200] # default xlim for the plots
        
        ####### fit stuff: deviation of the mean from the peak finding value and maxfev
        self.bound_mu        = 10 
        self.maxfev          = 10000
        
        ####### dictionarys for data, hist, fit
        self.data            = {}
        self.hist            = {}
        self.fit             = {}
        self.peaks           = {}
        self.peak_prop       = {}
        self.n_peaks         = {}
        
        self.threshold       = 980
        
        self.fit_dc_lower_range = {}
        self.fit_dc_upper_range = {}
        
        ####### select channel
        if channel==-1:
            print('All channels selected')
            self.ch          = -1
        elif (type(channel)!=int and type(channel)!=list) or ((type(channel)==int) and ((channel >=32) or (channel <=-2))) or ((type(channel)==list) and not (all(isinstance(i, int) for i in channel))):
            print('Weird channel selection ',channel)
            self.ch          = -1
        elif (type(channel)==int) and (channel!=-1):
            print(f'Selected channel: {str(i) for i in channel}')
            self.ch              = [channel]
        elif (type(channel)==list) and (all(isinstance(i, int) for i in channel)):
            print(f'Selected channel: {channel}')
            self.ch              = channel
        else:
            print('WTF, this channel selection should not happen!')
            
        
        if self.ch != -1:
            for i in self.ch:
                self.fit_dc_lower_range[i] = {}
                self.fit_dc_upper_range[i] = {}
        
        if self.ch == -1:
            for i in range(32):
                self.fit_dc_lower_range[i] = {}
                self.fit_dc_upper_range[i] = {}
        
        ####### file selection
        if file:
            self.file        = file
        else:
            print('No file selected')
    
    def load_data(self):
        try:
            data = np.genfromtxt(self.file)[1:,:]
            print('data loaded with numpy.genfromtxt')
        except:
            data = []
            with open(self.file) as f:
                lines = csv.reader((line.replace('\0','') for line in f), delimiter=' ')
                for row in lines:
                    try:
                        data.append(row)
                    except:
                        pass
            data = [[int(k) for k in j] for j in data[1:]]
            data = np.array(data[:-1])
            print('data not loaded with numpy.genfromtxt')
        if self.ch == -1:
            for i in range(32):
                self.data[i]  = {'hit':data[:,3*i],'lg':data[:,3*i+1],'hg':data[:,3*i+2]}
        else:
            for i in self.ch: # maybe change to range(32) because loading takes time, incase in hindsight one wants to analyse an additional channel
                self.data[i]  = {'hit':data[:,3*i],'lg':data[:,3*i+1],'hg':data[:,3*i+2]}
        self.data['temp'] = data[-1]
            
    def create_hist(self):
        # create the histogram from data
        # maybe plot??? idk
        if self.ch == -1:
            for i in range(32):
                self.hist[i]    = {}
                self.hist[i]['lg'], self.bin_edges = np.histogram(self.data[i]['lg'],bins=self.bins_edges) # exchanged self.bin_edges through _
                self.hist[i]['hg'], self.bin_edges = np.histogram(self.data[i]['hg'],bins=self.bins_edges)
                self.bin_center = [(self.bin_edges[i]+self.bin_edges[i+1])/2 for i in range(len(self.bin_edges)-1)]
        else:
            for i in self.ch:
                self.hist[i]    = {}
                self.hist[i]['lg'], self.bin_edges = np.histogram(self.data[i]['lg'],bins=self.bins_edges) # exchanged self.bin_edges through _
                self.hist[i]['hg'], self.bin_edges = np.histogram(self.data[i]['hg'],bins=self.bins_edges)
                self.bin_center = [(self.bin_edges[i]+self.bin_edges[i+1])/2 for i in range(len(self.bin_edges)-1)]
        
    def find_peaks(self):
        # find the peaks in the histogram
        if self.ch == -1:
            for i in range(32):
                self.peaks[i], self.peak_prop[i] = find_peaks(self.hist[i]['hg'],distance=self.peak_distance,prominence=self.peak_prominence,height=self.peak_height,width=self.peak_width)
                self.n_peaks[i]               = len(self.peaks[i])
                for j in range(self.n_peaks[i]):
                    self.fit_dc_lower_range[i][j] = 3
                    self.fit_dc_upper_range[i][j] = 10
                self.fit_dc_lower_range[i][0] = 10
                self.fit_dc_upper_range[i][0] = 20
        
        else:
            for i in self.ch:
                self.peaks[i], self.peak_prop[i] = find_peaks(self.hist[i]['hg'],distance=self.peak_distance,prominence=self.peak_prominence,height=self.peak_height,width=self.peak_width)
                self.n_peaks[i]               = len(self.peaks[i])
                for j in range(self.n_peaks[i]):
                    self.fit_dc_lower_range[i][j] = 3
                    self.fit_dc_upper_range[i][j] = 10
                self.fit_dc_lower_range[i][0] = 10
                self.fit_dc_upper_range[i][0] = 20
                
        
    
    
    def fit_peaks(self):
        # create p0 and bounds and fit peaks based on the find_peaks results
        self.fit['p0']     = {}
        self.fit['bounds'] = {}
        self.fit['popt']   = {}
        self.fit['pcov']   = {}
        
        if self.ch==-1:
            for i in range(32):
                p0 = [[self.peak_prop[i]['peak_heights'][j]/2,self.peaks[i][j],10] for j in range(len(self.peaks[i]))]

                bounds_lower = [[0,self.peaks[i][j]-self.bound_mu,0] for j in range(len(self.peaks[i]))]
                bounds_upper = [[np.inf,self.peaks[i][j]+self.bound_mu,20] for j in range(len(self.peaks[i]))]

                bounds_lower = [k for j in bounds_lower for k in j]
                bounds_upper = [k for j in bounds_upper for k in j]

                self.fit['p0'][i]     = [k for j in p0 for k in j]
                self.fit['bounds'][i] = [bounds_lower,bounds_upper]

                self.fit['popt'][i], self.fit['pcov'][i] = curve_fit(gauss,self.bin_center[self.fit_range[0]:self.fit_range[1]],self.hist[i]['hg'][self.fit_range[0]:self.fit_range[1]],
                                                 p0=self.fit['p0'][i],bounds=self.fit['bounds'][i], maxfev=self.maxfev)#_func[self.n_peaks]
                
        else:
            for i in self.ch:
                p0 = [[self.peak_prop[i]['peak_heights'][j]/2,self.peaks[i][j],10] for j in range(len(self.peaks[i]))]

                bounds_lower = [[0,self.peaks[i][j]-self.bound_mu,0] for j in range(len(self.peaks[i]))]
                bounds_upper = [[np.inf,self.peaks[i][j]+self.bound_mu,20] for j in range(len(self.peaks[i]))]

                bounds_lower = [k for j in bounds_lower for k in j]
                bounds_upper = [k for j in bounds_upper for k in j]

                self.fit['p0'][i]     = [k for j in p0 for k in j]
                self.fit['bounds'][i] = [bounds_lower,bounds_upper]

                self.fit['popt'][i], self.fit['pcov'][i] = curve_fit(gauss,self.bin_center[self.fit_range[0]:self.fit_range[1]],self.hist[i]['hg'][self.fit_range[0]:self.fit_range[1]],
                                                 p0=self.fit['p0'][i],bounds=self.fit['bounds'][i],maxfev=self.maxfev)#_func[self.n_peaks]
        
    def fit_peaks_dark_counts(self,verbose=False):
    # create p0 and bounds and fit peaks based on the find_peaks results
        self.fit['p0']     = {}
        self.fit['bounds'] = {}
        self.fit['popt']   = {}
        self.fit['pcov']   = {}

        if self.ch==-1:
            for i in range(32):
                p0 = [[self.peak_prop[i]['peak_heights'][j]/2,self.peaks[i][j],10] for j in range(len(self.peaks[i]))]

                bounds_lower = [[0,self.peaks[i][j]-self.bound_mu,0] for j in range(len(self.peaks[i]))]
                bounds_upper = [[np.inf,self.peaks[i][j]+self.bound_mu,20] for j in range(len(self.peaks[i]))]

                bounds_lower = [k for j in bounds_lower for k in j]
                bounds_upper = [k for j in bounds_upper for k in j]

                self.fit['p0'][i]     = [k for j in p0 for k in j]
                self.fit['bounds'][i] = [bounds_lower,bounds_upper]
                
                self.fit['popt'][i] = {}
                self.fit['pcov'][i] = {}

                for k in range(self.n_peaks[i]):
                    p0                  = self.fit['p0'][i][k*3:k*3+3]
                    bounds              = [self.fit['bounds'][i][0][k*3:k*3+3],self.fit['bounds'][i][1][k*3:k*3+3]]
                    self.fit['popt'][i][k], self.fit['pcov'][i][k] = curve_fit(gauss,self.bin_center[p0[1]-self.fit_dc_lower_range[i][k]:p0[1]+self.fit_dc_upper_range[i][k]],
                                                                               self.hist[i]['hg'][p0[1]-self.fit_dc_lower_range[i][k]:p0[1]+self.fit_dc_upper_range[i][k]],
                                                                               p0=p0,bounds=bounds, maxfev=self.maxfev)#_func[self.n_peaks]
                    if verbose:
                        print(f'Peak {k} fitted.')

        else:
            for i in self.ch:
                p0 = [[self.peak_prop[i]['peak_heights'][j]/2,self.peaks[i][j],10] for j in range(len(self.peaks[i]))]

                bounds_lower = [[0,self.peaks[i][j]-self.bound_mu,0] for j in range(len(self.peaks[i]))]
                bounds_upper = [[np.inf,self.peaks[i][j]+self.bound_mu,20] for j in range(len(self.peaks[i]))]

                bounds_lower = [k for j in bounds_lower for k in j]
                bounds_upper = [k for j in bounds_upper for k in j]

                self.fit['p0'][i]     = [k for j in p0 for k in j]
                self.fit['bounds'][i] = [bounds_lower,bounds_upper]
                
                self.fit['popt'][i] = {}
                self.fit['pcov'][i] = {}

                for k in range(self.n_peaks[i]):
                    p0                  = self.fit['p0'][i][k*3:k*3+3]
                    bounds              = [self.fit['bounds'][i][0][k*3:k*3+3],self.fit['bounds'][i][1][k*3:k*3+3]]
                    self.fit['popt'][i][k], self.fit['pcov'][i][k] = curve_fit(gauss,self.bin_center[p0[1]-self.fit_dc_lower_range[i][k]:p0[1]+self.fit_dc_upper_range[i][k]],
                                                                               self.hist[i]['hg'][p0[1]-self.fit_dc_lower_range[i][k]:p0[1]+self.fit_dc_upper_range[i][k]],
                                                                               p0=p0,bounds=bounds,maxfev=self.maxfev)#_func[self.n_peaks]
                    if verbose:
                        print(f'Peak {k} fitted.')
        
        
    def analyse(self,load = False):
        if load:
            self.load_data()
        self.create_hist()
        self.find_peaks()
        self.fit_peaks()
        
        
    def find_first_photon_number(self,cut_first=0,cut_last=0):
        if not self.noise_peak:
            self.noise_peak = 962
            print(f'Noise peak position assumed to be {self.noise_peak}')
        
        if cut_last == 0:
            self.gain_popt, self.gain_pcov = curve_fit(linear,np.linspace(cut_first,self.n_peaks-1,self.n_peaks-cut_first),np.reshape(self.popt,(self.n_peaks,3))[cut_first:,1],maxfev=100000)
        
        else:
            self.gain_popt, self.gain_pcov = curve_fit(linear,np.linspace(cut_first,self.n_peaks-1-cut_last,self.n_peaks-cut_last-cut_first),np.reshape(self.popt,(self.n_peaks,3))[cut_first:-cut_last,1],maxfev=100000)
        
        self.first_photon_number = round((self.gain_popt[1]-self.noise_peak)/self.gain_popt[0])
        
        self.gain                = self.gain_popt[0]
        self.noise_pos_from_fit  = linear(-self.first_photon_number,*self.gain_popt)
        
    
    def plot_data(self,save_plot=False,gain='hg',fit=True):
        
        if self.ch==-1:
            for i in range(32):
                fig, ax = plt.subplots(1,2)
                for j in ax:
                    j.hist(self.data[i][gain],bins=self.bins)
                    if fit:
                        j.plot(self.x,gauss(self.x,*self.fit['popt'][i]),'r-',lw=1)
                    j.set_xlim(self.xlim)
                    j.set_xlabel('Amplitude [ADCu]')
                    j.set_ylim(1,np.max(self.hist[i][gain])*1.1)
                ax[1].set_yscale('log')
                if save_plot!=False:
                    fig.savefig(save_plot)
                plt.show()
                
        else:
            for i in self.ch:
                fig, ax = plt.subplots(1,2)
                for j in ax:
                    j.hist(self.data[i][gain],bins=self.bins)
                    if fit:
                        j.plot(self.x,gauss(self.x,*self.fit['popt'][i]),'r-',lw=1)
                    j.set_xlim(self.xlim)
                    j.set_xlabel('Amplitude [ADCu]')
                    j.set_ylim(1,np.max(self.hist[i][gain])*1.1)
                ax[1].set_yscale('log')
                if save_plot!=False:
                    fig.savefig(save_plot)
                plt.show()
        
    def gain_plot(self,dc=False):
        # plot the mean of the peaks over the peak number and add the linear fit for the gain
        # add residuum plot
        
        fig, [ax,res] = plt.subplots(2,1)
        
        ax.plot(np.linspace(self.first_photon_number,np.peaks+self.first_photon_number,np.peaks-1+self.first_photon_number),self.popt[1::3])
        res.plot(np.linspace(self.first_photon_number,self.n_peaks-1+self.first_photon_number,self.n_peaks),self.popt[1::3]-linear(np.linspace(self.first_photon_number,self.n_peaks-1+self.first_photon_number,self.n_peaks),*self.gain_popt),'o',markersize=1)
        res.hlines(0,self.first_photon_number,self.n_peaks+self.first_photon_number,'r')
    
    def save(self,filename=False):
        if filename == False:
            filename = self.directory
        with open(filename+'array_save','wb') as f:
            pickle.dump(self,f)
    
    @staticmethod
    def load(filename):
        with open(filename+'array_save','rb') as f:
            return pickle.load(f)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     