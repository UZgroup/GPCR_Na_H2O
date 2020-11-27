#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:19:43 2019

@author: neilthomson


This script is designed to discretize the continuous distributions of 
any feature (torsion angles, interatomic distances, etc.) into well-defined 
distinct states. We use a Gaussian clustering algorithm to define states.

To fit the gaussians we need to obtain 3 guess parameters - sigma, mean, and amplitude

We do this by first correcting for the periodicity of the continuous distributions.
Angles can be measured based on pre-defined limits of e.g. -180 to 180 degrees, or 0 to 360. 
Periodic correction ensures that the histogram of the continuous distribution 
starts from a probability minima corresponding to the edge of a state. Periodic correction is different 
for water polarisation (spherical coordinate limits) and for torsions [0,360].
It is critical that the periodic correction parameters used to obtain the state limits
match the parameters used to periodically correct within the SSI transfer script. 

Following period correction, the histogram distributions are smoothed to ensure that any 
local maxima and minima corresponding to under-sampling are removed. The smoothing should be minimal
and is directly related to the histrogram bin number.

From the smoothed distributions, our algorithm locates all maxima. Any maxima 
with probabilities that are <0.75% of the global maxima are neglected. The remaining maxima are halfed,
and the y coordinates that have the closest to half maxima are selected. The number of closest y values 
can be modified in the [noc] parameter. We found that a value of 6 is sufficient for accurate clustering. 
We extract the corresponding x value closest to the x value of the maxima to obtain
a guess parameter for the sigma of that Gaussian. We therefore have the amplitudes, mean x, and sigma.

State limits are defined as the intersects of each gaussian distribution, and the 
periodic boundaries of the distribution. For water molecules, an additional state 
limit is added to determine whether the water is in the defined internal water pocket or not.

These state limits are ouput in filenames corresponding to the input. 
"""      

import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from queue import PriorityQueue 
import glob
import matplotlib.pyplot as plt
import re
from pathlib import Path

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'START OF FUNCTIONS'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"smoothing the kde data so that the extrema can be found without any of the small noise appearing as extrema"
def smooth(x,window_len,window='hanning'):
    if x.ndim != 1:
        raise ValueError
    if x.size < window_len:
        raise ValueError
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

'FINDING THE NEAREST NEIGHBOUR FUNCTION'
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

'GAUSSIAN FUNCTIONS'
def gauss(x, x0, sigma, a):
    """ Gaussian function: """
    return abs(a*np.exp(-(x-x0)**2/(2*sigma**2)))
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    """ Two gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
    """ Three gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)
def quadmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4):
    """ Four gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)
def quinmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5):
    """ Five gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)
def sexmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6):
    """ Six gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)   
def septmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7):
    """ Six gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)    
'CHECK IF VALUE IS BETWEEN X AND Y'
def check(value,x,y):
    if x <= value <= y:
        return 1
    else:
        return 0

'PRINT K CLOSEST VALUES TO SPECIFIED VALUE'
def printKclosest(arr,n,x,k): 
    a=[]
  
    # Make a max heap of difference with  
    # first k elements.  
    pq = PriorityQueue() 
    for i in range(k): 
        pq.put((-abs(arr[i]-x),i)) 
  
    # Now process remaining elements 
    for i in range(k,n): 
        diff = abs(arr[i]-x) 
        p,pi = pq.get() 
        curr = -p 
  
        # If difference with current  
        # element is more than root,  
        # then put it back.  
        if diff>curr: 
            pq.put((-curr,pi)) 
            continue
        else: 
  
            # Else remove root and insert 
            pq.put((-diff,i)) 
              
    # Print contents of heap. 
    while(not pq.empty()): 
        p,q = pq.get() 
        a.append(str("{} ".format(arr[q])))
    
    return a

##correcting for the periodicity of torsions
def periodic_correction(angle1):

    ##generating a histogram of the chi angles
    heights=np.histogram(angle1, bins=90, density=True)
    if show_periodic_correction_plots=='yes':
        plt.figure()
        plt.plot(heights[1][0:-1],heights[0])
        plt.title('pre periodic correction')
    ##if the first bar height is greater than the minimum cut off
    ##then find the smallest bar and shift everything before that bar by 360
#    if heights[0][0] != min(heights[0]):   
    if heights[0][0] > max(heights[0])*0.0005:   

        ##set the periodic boundary to the first minimum in the distribution
        ##find the minimum angle by multiplying thre minimum bin number by the size of the bins
        ##define the new periodic boundary for the shifted values
        j=np.where(heights[0] == min(heights[0]))[0][0]*(360.0/len(heights[0]))-180
        ##remove the -180 for waters since they by default range from 0 to 360

#        print(j)
        for k in range(len(angle1)):
            ##if the angle is before the periodic boundary, shift by 360
            if angle1[k] <= j:
                angle1[k]+=360
                
    if show_periodic_correction_plots=='yes':         
        plt.figure()
        plt.hist(angle1,bins=90)
        plt.title('post periodic correction')

    return angle1

##correcting for the periodicity of waters
def periodic_correction_h2o(angle1):

    ## 10000.0 is the value corresponding to an unoccupied water pocket, we
    ## remove these in order to correct the periodicity of just the continuous 
    ## distribution of water polarisation
    dipole_angle = [i for i in angle1 if i != 10000.0]
    # print(len(dipole_angle))
    indices = [i for i, x in enumerate(angle1) if x != 10000.0]

    heights=np.histogram(dipole_angle, bins=90, density=True)
    if show_periodic_correction_plots=='yes':
        plt.figure()
        plt.plot(heights[1][0:-1],heights[0])
        plt.title('pre periodic correction')
        
    ##if the first bar height is greater than the minimum cut off
    ##then find the smallest bar and shift everything before that bar by 360
    if heights[0][0] > max(heights[0])*0.0005:   

        ##set the periodic boundary to the first minimum in the distribution
        ##find the minimum angle by multiplying thre minimum bin number by the size of the bins
        ##define the new periodic boundary for the shifted values
        j=np.where(heights[0] == min(heights[0]))[0][0]*(360.0/len(heights[0]))
        for k in range(len(dipole_angle)):
            ##if the angle is before the periodic boundary, shift by 360
            if dipole_angle[k] <= j:
                dipole_angle[k]+=360
    
    for i in range(len(indices)):
        angle1[indices[i]] = dipole_angle[i]
        
    if show_periodic_correction_plots=='yes':         
        plt.figure()
        plt.hist(angle1,bins=90)
        plt.title('post periodic correction')

    return angle1


## obatining the gaussians that fit the distribution
def get_gaussian_fit(distribution,resname):
    
    histo=np.histogram(distribution, bins=binnumber, density=True)
    distributionx=smooth(histo[1][0:-1],window_len)
    distributiony=smooth(histo[0]-min(histo[0]),window_len)
    
    ##getting an array of all the maxima indices
    maxima = [distributiony[item] for item in argrelextrema(distributiony, np.greater)][0]

    ##the maxima may be an artifact of undersampling
    ##this grabs only the maxima that correspond to a density greater than the cutoff
    ##cutoff= 0.75% at a 99.25% significance level
    ##which ignores only the states limits in which states are sampled less that 0.75% of the time
    corrected_extrema=[item for item in maxima if item > max(distributiony)*0.0075]
    
    ##finding the guess parameters for the plots
    ##empty lists for the guess params to be added to
    mean_pop=[]
    sigma_pop=[]
    
    ##number of closest neighbours
    ##setting for the sigma finding function
    noc=6
    
    ##for all the extrema, find the 'noc' closest x coordinates that lie on the distribution
    ##closest to a value of half of the maximum
    ##this is to find the edges of the gaussians for calculating sigma
    sig_vals=[]
    for extrema in corrected_extrema:
        ##finding the 6 y values closest to the half max value of each extrema
        closest=printKclosest(distributiony, len(distributiony), extrema/2.0, noc)
        ##finding the x coordinate corresponding to these values
        xlist=[np.where(distributiony==float(closesty))[0][0] for closesty in closest]
        xsig=find_nearest(distributionx[xlist],distributionx[np.where(distributiony==extrema)[0][0]])
        ##obtaining the width of the distribution
        sig=np.absolute(xsig-distributionx[np.where(distributiony==extrema)[0][0]])
#        print(xlist,xsig,sig)
        sig_vals.append(sig)
        
    ##the mean x of the gaussian is the value of x at the peak of y
    mean_vals=[distributionx[np.where(distributiony==extrema)[0][0]] for extrema in corrected_extrema]
    for i in range(len(corrected_extrema)):
        mean_pop.append(mean_vals[i])
        sigma_pop.append(sig_vals[i])


    ##x is the space of angles
    xline=np.linspace(min(distribution),min(distribution)+360,10000)                
    
    ##choosing the fitting mode
    peak_number=[gauss,bimodal,trimodal,quadmodal,quinmodal,sexmodal,septmodal]
    mode=peak_number[len(sig_vals)-1]
    
    expected=[]
    for i in range(len(mean_pop)):
        expected.append(mean_pop[i])
        expected.append(sigma_pop[i])
        expected.append(corrected_extrema[i])
    
    params,cov=curve_fit(mode,distributionx,distributiony,expected)
   
    if show_gaussian_plots=='yes':
        plt.figure()
        sns.distplot(distribution,bins=binnumber) 
        plt.title(resname)
    gaussians=[]
    colours=['m','g','c','r','b','y','k']
    gaussnumber=np.linspace(0,(len(params))-3,int(len(params)/3))
    
    for j in gaussnumber:
        gaussians.append(gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]))
        if show_gaussian_plots=='yes':
            plt.plot(xline,gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]),
                      color=colours[np.where(gaussnumber==j)[0][0]], linewidth=2)

    return gaussians, xline


'OBTAINING THE GAUSSIAN INTERSECTS'
def get_intersects(gaussians,distribution,xline):
    ##discretising each state by gaussian intersects
    
    ##adding the minimum angle value as the first boundary
    all_intersects=[min(distribution)]
    for i in range(len(gaussians)-1):
        
        ##First calculate f - g and the corresponding signs using np.sign. 
        ##Applying np.diff reveals all the positions where the sign changes (e.g. the lines cross). 
        ##Using np.argwhere gives us the exact indices of the state intersects
        idx = np.argwhere(np.diff(np.sign(gaussians[i] - gaussians[i+1]))).flatten()
        for intersect in idx:
            all_intersects.append(xline[intersect])
            if show_gaussian_plots=='yes':
                plt.axvline(all_intersects[-1],color='k',lw=1,ls='--')
        
    all_intersects.append(max(distribution))
    return all_intersects
    
##this function requires a list of the distribution you want to cluster/discretize into states
##this can be applied to a list of all filenames in a directory where every filename is a list of the distributions
def extract_state_limits(resid):

    "WATERS"
    # files1 = [f.split("chi1/")[1] for f in glob.glob("prot/waters/chi1/" + "**/*.txt", recursive=True)]
    # files2 = [f.split("chi1/")[1] for f in glob.glob("unprot/waters/chi1/" + "**/*.txt", recursive=True)]
    # file_list=[files1,files2]
    
    "RESIDUES"
    files1 = [f.split("rama/")[1] for f in glob.glob("prot/rama/" + "**/*.xvg", recursive=True)]
    files2 = [g.split("rama/")[1] for g in glob.glob("unprot/rama/" + "**/*.xvg", recursive=True)]
    file_list=[files1,files2]
    files_ordered=[[],[]]
    
    ##a list of the residue number for the ACE terminal of the gpcr
    ##parsing the data by separating the residue number and re-ordering the lists numerically
    for i in range(len(files_ordered)):
        for r in file_list[i]:
            j=re.split('(\d+)',r)
            files_ordered[i].append(j)
        files_ordered[i].sort(key = lambda x: int(x[1]))
        
    ##adding together the strings to create one string
    for i in range(len(files_ordered)):
        for j in range(len(files_ordered[i])):
            files_ordered[i][j]=files_ordered[i][j][0]+files_ordered[i][j][1]+files_ordered[i][j][2]    
    
    "WATERS"
    # gro_res_list=range(len(file_list[0]))
    # names=file_list[0]
    
    "RESIDUES"
    residue_list=range(len(files_ordered[0]))
    names=files_ordered[0]    

    residue=residue_list.index(resid)
    resname=names[residue]
    state_limits=[]    
    intersection_of_states=[]

    print('Importing the data')
    # "UNCOMMENT FOR WATER STATE ANALYSIS"
    # ##the list is cropped so the simulations are the same lengths
    # sim1_X1 = list(np.genfromtxt("unprot/waters/chi1/" + names[residue]))
    # sim1_X2 = list(np.genfromtxt("unprot/waters/chi2/" + names[residue]))
    # sim2_X1 = list(np.genfromtxt("prot/waters/chi1/" + names[residue]))[0:len(sim1_X1)]
    # sim2_X2 = list(np.genfromtxt("prot/waters/chi2/" + names[residue]))[0:len(sim1_X1)]
    
    
    
    "UNCOMMENT FOR RESIDUE STATE ANALYSIS"

  
    sim1_X1 = [item[0] for item in list(np.genfromtxt("prot/rama/" + names[residue]))]
    my_file = Path("prot/rama/" + names[residue])
    if my_file.is_file():
        sim1_X2 = [item[1] for item in list(np.genfromtxt("prot/rama/" + names[residue]))]

    sim2_X1 = [item[0] for item in list(np.genfromtxt("unprot/rama/" + names[residue]))][0:len(sim1_X1)]
    #import chi2 file if it exists
    my_file = Path("unprot/rama/" + names[residue])
    if my_file.is_file():
        sim2_X2 = [item[1] for item in list(np.genfromtxt("unprot/rama/" + names[residue]))][0:len(sim1_X1)]
        
        
    print('# of frames sim 1', len(sim1_X1))
    print('# of frames sim 2', len(sim2_X1))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""          
    "OBTAINING BIVARIATE DIHEDRAL STATE"

    "UNCOMMENT FOR WATER STATE ANALYSIS"
    # h2o_dists=sim1_X1+sim2_X1
    # dist=[item for item in h2o_dists if item != 10000.0]
    
    # chi_of_interest=periodic_correction_h2o(dist)   
    # gaussians, xline = get_gaussian_fit(chi_of_interest,resname)            
    # ##discretising each state by gaussian intersects       
    # h2o_states=[]
    # h2o_states.append(get_intersects(gaussians,chi_of_interest,xline))
    # h2o_states[-1].append(20000.0)
    # print(h2o_states)
    # intersection_of_states.append(h2o_states[0])  
    
    
    # h2o_dists=sim1_X2+sim2_X2
    # dist=[item for item in h2o_dists if item != 10000.0]
    # chi_of_interest=periodic_correction_h2o(dist)   
    # gaussians, xline = get_gaussian_fit(chi_of_interest,resname)            
    # ##discretising each state by gaussian intersects       
    # h2o_states=[]
    # h2o_states.append(get_intersects(gaussians,chi_of_interest,xline))
    # h2o_states[-1].append(20000.0)
    # print(h2o_states)
    # intersection_of_states.append(h2o_states[0])     
    """"""""

    "UNCOMMENT FOR RESIDUE STATE ANALYSIS"
    #adding the chi angles for a single residue together for all simulation
    #and correcting the periodic boundary conditions
    chi_of_interest = periodic_correction(sim1_X1+sim2_X1)              
    ##obtaining the gaussian fit
    gaussians, xline = get_gaussian_fit(chi_of_interest,resname)            
    ##discretising each state by gaussian intersects       
    intersection_of_states.append(get_intersects(gaussians,chi_of_interest,xline))   
    ##checking if there is a chi2 to analyse also
    if my_file.is_file():
        chi_of_interest = periodic_correction(sim1_X2+sim2_X2)          
        gaussians, xline = get_gaussian_fit(chi_of_interest,resname)
        intersection_of_states.append(get_intersects(gaussians,chi_of_interest,xline))   
    """"""""

    ##if there are two chi values, then append them in the correct order to state limits list
    if len(intersection_of_states)>1: 
        state_limits.append(intersection_of_states[-2])
    state_limits.append(intersection_of_states[-1])
    


    resstates=intersection_of_states[-2:]
    print(resstates)
    # filename='state_intersects/'+file_list[0][residue][:-4]+'state_intersects.txt'
        
    filename='state_intersects/'+files_ordered[0][residue][:-4]+'state_intersects.txt'

    print('SAVED ', filename)
    print('FINISHED RESIDUE ', residue)
    
    # with open(filename, 'w') as output:
    #     for row in resstates:
    #         output.write(str(row) + '\n')
        
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'END OF FUNCTIONS'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Below are the inputs for the state finding functions. 
To find the state limits of a feature, we need to input that specific feature -
e.g.

extract_state_limits(distribution)

The code is currently written to extract the state limits for an entire folder 
of residues corresponding to the whole protein. The input is an index which relates
to a list of filenames in the folder. The file is then imported, discretised, and the state_limits
are saved in another folder.

"""

##bin number for the gaussian fitting algorithm
##this determines the resolution of the distribution that the gaussians are fit to
##it may need to be altered to account for subtle differences in each distrbituion
binnumber=60
##the window length is the number of bins that are smoothed over, the binnumber and window length 
##should be altered together as more bins will likely require a larger window length
window_len=10


# "WATERS"
# files1 = [f.split("chi1/")[1] for f in glob.glob("prot/waters/chi1/" + "**/*.txt", recursive=True)]
# files2 = [f.split("chi1/")[1] for f in glob.glob("unprot/waters/chi1/" + "**/*.txt", recursive=True)]
# file_list=[files1,files2]
# gro_res_list=range(len(file_list[0]))
# names=file_list[0]

"RESIDUES"
files1 = [f.split("rama/")[1] for f in glob.glob("prot/rama/" + "**/*.xvg", recursive=True)]
files2 = [g.split("rama/")[1] for g in glob.glob("unprot/rama/" + "**/*.xvg", recursive=True)]
file_list=[files1,files2]
files_ordered=[[],[]]
##parsing the data by separating the residue number and re-ordering the lists numerically
for i in range(len(files_ordered)):
    for r in file_list[i]:
        j=re.split('(\d+)',r)
        files_ordered[i].append(j)
    files_ordered[i].sort(key = lambda x: int(x[1]))
    
##adding together the strings to create one string
for i in range(len(files_ordered)):
    for j in range(len(files_ordered[i])):
        files_ordered[i][j]=files_ordered[i][j][0]+files_ordered[i][j][1]+files_ordered[i][j][2]

residue_list=range(len(files_ordered[0]))
names=files_ordered[0]    

##show the plots for the periodic correction
show_periodic_correction_plots='no'
##show the plots to visualise the state discretization
show_gaussian_plots='yes'
