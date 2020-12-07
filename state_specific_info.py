#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:19:43 2020

@author: neilthomson
"""

import numpy as np
from queue import PriorityQueue 
import math
import re
import glob
import itertools
#from tqdm import tqdm
from multiprocessing import Pool
from time import gmtime, strftime
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

"""
FUNCTIONS
"""

#CHECK IF VALUE IS BETWEEN X AND Y
def check(value,x,y):
    if x <= value <= y:
        return 1
    else:
        return 0


#CORRECTING FOR THE PERIODICITY OF ANGLES
def periodic_correction(angle1):
    ##generating a histogram of the chi angles
    heights=np.histogram(angle1, bins=90, density=True)
    ##if the first bar height is greater than the minimum cut off
    ##then find the smallest bar and shift everything before that bar by 360
    if heights[0][0] > max(heights[0])*0.0005:   
        ##set the periodic boundary to the first minimum in the distribution
        ##find the minimum angle by multiplying thre minimum bin number by the size of the bins
        ##define the new periodic boundary for the shifted values
        ##ned to subtract 180 as distributions go [-180,180]
        j=np.where(heights[0] == min(heights[0]))[0][0]*(360.0/len(heights[0]))-180
        for k in range(len(angle1)):
            ##if the angle is before the periodic boundary, shift by 360
            if angle1[k] <= j:
                angle1[k]+=360
    return angle1


#CORRECTING FOR THE PERIODICITY OF WATER ANGLES
def periodic_correction_h2o(angle1):
    ##angle positions
    dipole_angle = [i for i in angle1 if i != 10000.0]
    indices = [i for i, x in enumerate(angle1) if x != 10000.0]
    ##generating a histogram of the chi angles
    heights=np.histogram(dipole_angle, bins=90, density=True)
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
    return angle1


def import_distribution(simulation_folder, file_name):
    dist1 = [item[0] for item in list(np.genfromtxt(simulation_folder + file_name))]
    my_file = Path(simulation_folder + file_name)
    if my_file.is_file():
        dist2 = [item[1] for item in list(np.genfromtxt(simulation_folder + file_name))]
        return dist1, dist2
    else:
        return dist1
    
    
# this function makes sure that the two simulations are the same length
def match_sim_lengths(sim1,sim2):
    if len(sim1)!=len(sim2):
        if len(sim1)>len(sim2):
            sim1=sim1[0:len(sim2)]
        if len(sim1)<len(sim2):
            sim2=sim2[0:len(sim1)]  
    return sim1, sim2
        

def get_filenames(folder):  
    files = [f.split(folder)[1] for f in glob.glob(folder+ "**/*.xvg", recursive=True)]
    files_ordered=[]
    ##parsing the data by separating the residue number and re-ordering the lists numerically
    for r in files:
        j=re.split('(\d+)',r)
        files_ordered.append(j)
    files_ordered.sort(key = lambda x: int(x[1]))
    ##adding together the strings to create one string
    for i in range(len(files_ordered)):
        files_ordered[i]=files_ordered[i][0]+files_ordered[i][1]+files_ordered[i][2]
    return files_ordered
    

#smoothing the kde data so that the extrema can be found without any of the small noise appearing as extrema
def smooth(x,window_len,window=None):
    if window is None:
        window_type='hanning'
    if x.ndim != 1:
        raise ValueError
    if x.size < window_len:
        raise ValueError
    if window_len<3:
        return x
    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window_type  is  'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window_type+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


#FINDING THE NEAREST NEIGHBOUR FUNCTION
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


#GAUSSIAN FUNCTIONS
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
    """ Seven gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)    


#PRINT K CLOSEST VALUES TO SPECIFIED VALUE
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


## obatining the gaussians that fit the distribution
def get_gaussian_fit(distribution, binnumber=60, window_len=10, show_plots=None):
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
    if show_plots is not None:
        plt.figure()
        sns.distplot(distribution,bins=binnumber) 
    gaussians=[]
    colours=['m','g','c','r','b','y','k']
    gaussnumber=np.linspace(0,(len(params))-3,int(len(params)/3))    
    for j in gaussnumber:
        gaussians.append(gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]))
        if show_plots is not None:
            plt.plot(xline,gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]),
                      color=colours[np.where(gaussnumber==j)[0][0]], linewidth=2)
    return gaussians, xline


# OBTAINING THE GAUSSIAN INTERSECTS
def get_intersects(gaussians,distribution,xline, show_plots=None):
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
    all_intersects.append(max(distribution))        
    if show_plots is not None:
        plt.figure()
        sns.distplot(distribution,bins=90) 
        for i in range(len(all_intersects)):
            plt.axvline(all_intersects[i],color='k',lw=1,ls='--')    
    return all_intersects
    

##this function requires a list of the distribution you want to cluster/discretize into states
##this can be applied to a list of all filenames in a directory where every filename is a list of the distributions
def extract_state_limits(distr, show_plots=None):    
    ##obtaining the gaussian fit
    gaussians, xline = get_gaussian_fit(distr)            
    ##discretising each state by gaussian intersects       
    intersection_of_states=get_intersects(gaussians,distr,xline,show_plots)   
    return intersection_of_states

def calculate_entropy(state_limits,distribution_list):
    ## subtract 1 since number of partitions = number of states - 1
    mut_prob=np.zeros(([len(state_limits[i])-1 for i in range(len(state_limits))]))     
    ##obtaining the entropy
    entropy=0
    ##iterating over every multidimensional index in the array
    it = np.nditer(mut_prob, flags=['multi_index'])
    while not it.finished:
        ##grabbing the indices of each element in the matrix
        arrayindices=list(it.multi_index)
        ##making an array of the state occupancy of each distribution
        limit_occupancy_checks=np.zeros((len(arrayindices), len(distribution_list[0])))
        for i in range(len(arrayindices)):
            ##obtaining the limits of each state 
            ##in the identified array position            
            limits=[state_limits[i][arrayindices[i]],state_limits[i][arrayindices[i]+1]]
            ##grabbing the distribution that the limits corresponds to
            distribution=distribution_list[i]
            ##checking the occupancy of this state with the distribution
            for j in range(len(distribution)):
                limit_occupancy_checks[i][j]=check(distribution[j],limits[0],limits[1]) 
        ##calculating the probability as a function of the occupancy
        mut_prob[it.multi_index]=sum(np.prod(limit_occupancy_checks,axis=0)) / len(limit_occupancy_checks[0])
        ##calculating the entropy as the summation of all -p*log(p) 
        if mut_prob[it.multi_index] != 0:
            entropy+=-1*mut_prob[it.multi_index]*math.log(mut_prob[it.multi_index],2)
        it.iternext()
    return entropy

##this function requires a list of angles for SSI
##SSI(A,B) = H(A) + H(B) - H(A,B)
def calculate_ssi(set_distr_a, set_distr_b=None):
    
    ##calculating the entropy for set_distr_a
    ## if set_distr_a only contains one distributions
    if any(isinstance(i, list) for i in set_distr_a) is 0:
        distr_a=[periodic_correction(set_distr_a)]
    ## else set_distr_a is a nested list of multiple distributions (bivariate)
    else:
        distr_a=[periodic_correction(i) for i in set_distr_a]
    distr_a_states=[]
    for i in distr_a:
        distr_a_states.append(extract_state_limits(i,show_plots=True))
    H_a=calculate_entropy(distr_a_states,distr_a)
    
    ##calculating the entropy for set_distr_b
    ## if no dist (None) then apply the binary dist for two simulations
    if set_distr_b is None:       
        H_b=1
        distr_b=[[0.5]*int(len(distr_a[0])/2) + [1.5]*int(len(distr_a[0])/2)]
        distr_b_states= [[0,1,2]]  
        
    else:
        if any(isinstance(i, list) for i in set_distr_b) is 0:
            distr_b=[periodic_correction(set_distr_b)]
        else:
            distr_b=[periodic_correction(i) for i in set_distr_b]
        distr_b_states=[]
        for i in distr_b:
            distr_b_states.append(extract_state_limits(i))
        H_b=calculate_entropy(distr_b_states,distr_b)



    print(distr_a_states)
    print(distr_b_states)
    ab_joint_states= distr_a_states + distr_b_states
    ab_joint_distributions= distr_a + distr_b
    print(ab_joint_states)
    
    H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)

    SSI = (H_a + H_b) - H_ab
        
    return SSI


#CoSSI = H_a + H_b + H_c - H_ab - H_bc - H_ac + H_abc
def calculate_cossi(set_distr_a, set_distr_b, set_distr_c=None):
    
    ##calculating the entropy for set_distr_a
    if sum(1 for x in set_distr_a if isinstance(x, list)) is 0:
        distr_a=[set_distr_a]
    else:
        distr_a=[i for i in set_distr_a]
    distr_a_states=[]
    for i in distr_a:
        distr_a_states.append(extract_state_limits(i))
    H_a=calculate_entropy(distr_a_states,distr_a)
        
    ##----------------
    ##calculating the entropy for set_distr_b
    if sum(1 for x in set_distr_b if isinstance(x, list)) is 0:
        distr_b=[set_distr_b]
    else:
        distr_b=[i for i in set_distr_b]
    distr_b_states=[]
    for i in distr_b:
        distr_b_states.append(extract_state_limits(i))
    H_b=calculate_entropy(distr_b_states,distr_b) 
    
    ##----------------
    ##calculating the entropy for set_distr_c
    ## if no dist (None) then apply the binary dist for two simulations
    if set_distr_c is None:
        H_c=1
        distr_c=[[0.5]*int(len(distr_a[0])/2) + [1.5]*int(len(distr_a[0])/2)]
        distr_c_states= [[0,1,2]]  
    else:
        if any(isinstance(i, list) for i in set_distr_c) is 0:
            distr_c=[periodic_correction(set_distr_c)]
        else:
            distr_c=[periodic_correction(i) for i in set_distr_c]
        distr_c_states=[]
        for i in distr_c:
            distr_c_states.append(extract_state_limits(i))
        H_c=calculate_entropy(distr_c_states,distr_c)

    ##----------------
    ab_joint_states= distr_a_states + distr_b_states
    ab_joint_distributions= distr_a + distr_b
    
    H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)
    ##----------------
    ac_joint_states= distr_a_states + distr_c_states 
    ac_joint_distributions= distr_a + distr_c
    
    H_ac= calculate_entropy(ac_joint_states,ac_joint_distributions)
    ##----------------
    bc_joint_states= distr_b_states + distr_c_states 
    bc_joint_distributions= distr_b + distr_c
    
    H_bc= calculate_entropy(bc_joint_states,bc_joint_distributions)
    ##----------------
    abc_joint_states= distr_a_states + distr_b_states + distr_c_states 
    abc_joint_distributions= distr_a + distr_b + distr_c
    
    H_abc=calculate_entropy(abc_joint_states,abc_joint_distributions)    
    
    
    SSI = (H_a + H_b) - H_ab
    coSSI = (H_a + H_b + H_c) - (H_ab + H_ac + H_bc) + H_abc 
        
    return SSI, coSSI

    
    
    
