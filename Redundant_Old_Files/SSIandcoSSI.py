#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:19:43 2020

@author: neilthomson
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##THIS SCRIPT WAS USED TO CALCULATE THE FINAL RESULTS FOR THE PAPER
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
#import seaborn as sns
#import matplotlib.pyplot as plt
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'FUNCTIONS'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

'CHECK IF VALUE IS BETWEEN X AND Y'
def check(value,x,y):
    if x <= value <= y:
        return 1
    else:
        return 0

'CORRECTING FOR THE PERIODICITY OF ANGLES'
def periodic_correction(angle1):

    ##generating a histogram of the chi angles
    heights=np.histogram(angle1, bins=90, density=True)

    ##if the first bar height is greater than the minimum cut off
    ##then find the smallest bar and shift everything before that bar by 360
#    if heights[0][0] != min(heights[0]):   
    if heights[0][0] > max(heights[0])*0.0005:   

        ##set the periodic boundary to the first minimum in the distribution
        ##find the minimum angle by multiplying thre minimum bin number by the size of the bins
        ##define the new periodic boundary for the shifted values
        j=np.where(heights[0] == min(heights[0]))[0][0]*(360.0/len(heights[0]))-180
        for k in range(len(angle1)):
            ##if the angle is before the periodic boundary, shift by 360
            if angle1[k] <= j:
                angle1[k]+=360

    return angle1

'CORRECTING FOR THE PERIODICITY OF WATER ANGLES'
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

def calculate_entropy(state_limits,distribution_list):
    
    ## subtract 1 since final state has two limits.
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
    
##this function requires a list for chi1 and chi2 dihedral angles 
def extract_inf_mat_row(pair_of_residues):

    files1 = [f.split("chi1/")[1] for f in glob.glob("definitive_results/naallchis/chi1/" + "**/*.xvg", recursive=True)]
    files2 = [g.split("chi1/")[1] for g in glob.glob("definitive_results/nonaallchis/chi1/" + "**/*.xvg", recursive=True)]
    files3 = [h.split("chi1/")[1] for h in glob.glob("definitive_results/protallchis/chi1/" + "**/*.xvg", recursive=True)]

    file_list=[files1,files2,files3]
    files_ordered=[[],[],[]]
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
            
    water_list=['n150.txt','n749.txt','c647.txt','w450.txt','d349.txt']
    for i in water_list:
        files_ordered[0].insert(1,i)
            
    res1=pair_of_residues[0]
    res2=pair_of_residues[1]
    gro_res_list=range(len(files_ordered[0])-2)
    names=files_ordered[0]
    
    selection1=gro_res_list.index(res1)
    selection2=gro_res_list.index(res2)

    "choose a set of microswitches to study the communication between them"
    info_transfer=[selection1,selection2]
    all_states=[]
    all_chis=[]
    
    entropies=[]

    noh2oentropies=[]
    noh2olimits=[]
    noh2odists=[]
    
    for residue in info_transfer:
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""             
        ## importing the data
        print('Importing the data for ' + names[residue][0:-4])
        ##resetting the bool for if there is a chi2 dimension                    
        chi2_bool = 0

        nadisth2opsi=list(np.genfromtxt('definitive_results/waterallchis/na/chi1/'+water_pocket_site+'.txt'))
        nadisth2ophi=list(np.genfromtxt('definitive_results/waterallchis/na/chi2/'+water_pocket_site+'.txt'))
        nonadisth2opsi=list(np.genfromtxt('definitive_results/waterallchis/nona/chi1/'+water_pocket_site+'.txt'))
        nonadisth2ophi=list(np.genfromtxt('definitive_results/waterallchis/nona/chi2/'+water_pocket_site+'.txt'))
        protdisth2opsi=list(np.genfromtxt('definitive_results/waterallchis/prot/chi1/'+water_pocket_site+'.txt'))
        protdisth2ophi=list(np.genfromtxt('definitive_results/waterallchis/prot/chi2/'+water_pocket_site+'.txt'))
        
        "Importing The Angle Distributions"
        ##the waters are located in a different folder
        if 0 < residue < 6:
            na_X1=list(np.genfromtxt('definitive_results/waterallchis/na/chi1/'+names[residue]))
            na_X2=list(np.genfromtxt('definitive_results/waterallchis/na/chi2/'+names[residue]))
            nona_X1=list(np.genfromtxt('definitive_results/waterallchis/nona/chi1/'+names[residue]))
            nona_X2=list(np.genfromtxt('definitive_results/waterallchis/nona/chi2/'+names[residue]))
            prot_X1=list(np.genfromtxt('definitive_results/waterallchis/prot/chi1/'+names[residue]))
            prot_X2=list(np.genfromtxt('definitive_results/waterallchis/prot/chi2/'+names[residue]))
            chi2_bool = 1

        else:                
            na_X1 = [item[1] for item in np.genfromtxt("definitive_results/naallchis/chi1/" + names[residue])]
            ##import chi2 file if it exists
            my_file = Path("definitive_results/naallchis/chi2/" + names[residue])
            if my_file.is_file():
                na_X2 = [item[1] for item in np.genfromtxt("definitive_results/naallchis/chi2/" + names[residue])]
                chi2_bool = 1
                
            nona_X1 = [item[1] for item in np.genfromtxt("definitive_results/nonaallchis/chi1/" + names[residue])]
            my_file = Path("definitive_results/nonaallchis/chi2/" + names[residue])
            if my_file.is_file():
                nona_X2 = [item[1] for item in np.genfromtxt("definitive_results/nonaallchis/chi2/" + names[residue])]
                chi2_bool = 1

                
            prot_X1 = [item[1] for item in np.genfromtxt("definitive_results/protallchis/chi1/" + names[residue])]
            my_file = Path("definitive_results/protallchis/chi2/" + names[residue])
            if my_file.is_file():
                prot_X2 = [item[1] for item in np.genfromtxt("definitive_results/protallchis/chi2/" + names[residue])]
                chi2_bool = 1
                

        "Importing The State Limits"
        if simulation_pair=='na_nona':
            filename='definitive_results/state_intersects/na_nona/'+names[residue][0:-4]+'state_intersects.txt'
        else:
            filename='definitive_results/state_intersects/nona_prot/'+names[residue][0:-4]+'state_intersects.txt'
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""     
        ## storing the dihedral state intersects for the mutual probability calculations
        current_states=[]
        current_dists=[]
        
        'Parsing the distributions and intersects of the simulations'
        
        if chi2_bool == 1:
#        if 'nona_X2' in locals():
          
            "OBTAINING BIVARIATE DIHEDRAL STATE"
            ##adding the chi angles together for simulations of interest
            ##and correcting the periodic boundary conditions
            if residue==0:
                ##don't coorect for periodicity in the input info
                if simulation_pair=='na_nona':
                    chi_of_interest=[na_X1 + nona_X1, na_X2 + nona_X2]
                else:
                    chi_of_interest=[nona_X1 + prot_X1, nona_X2 + prot_X2]
    
            elif 0 < residue < 6:
                ##only correct for periodicity in the frames where water is in the pocket
                if simulation_pair=='na_nona':
                    chi_of_interest = [periodic_correction_h2o(na_X1+nona_X1), periodic_correction_h2o(na_X2+nona_X2)]
                else:
                    chi_of_interest = [periodic_correction_h2o(nona_X1+prot_X1), periodic_correction_h2o(nona_X2+prot_X2)]
    
            else:
                ##always account for perdiocity in the dihedral angles
                if simulation_pair=='na_nona':
                    chi_of_interest = [periodic_correction(na_X1+nona_X1), periodic_correction(na_X2+nona_X2)]
                else:
                    chi_of_interest = [periodic_correction(nona_X1+prot_X1), periodic_correction(nona_X2+prot_X2)]

        else:
            
            "OBTAINING BIVARIATE DIHEDRAL STATE"
            ##adding the chi angles together for simulations of interest
            ##and correcting the periodic boundary conditions
            if residue==0:
                ##don't coorect for periodicity in the input info
                if simulation_pair=='na_nona':
                    chi_of_interest=[na_X1 + nona_X1]
                else:
                    chi_of_interest=[nona_X1 + prot_X1]
    
            elif 0 < residue < 6:
                ##only correct for periodicity in the frames where water is in the pocket
                if simulation_pair=='na_nona':             
                    chi_of_interest = [periodic_correction_h2o(na_X1+nona_X1)]
                else:
                    chi_of_interest = [periodic_correction_h2o(nona_X1+prot_X1)]
    
            else:
                ##always account for perdiocity in the dihedral angles
                if simulation_pair=='na_nona':
                    chi_of_interest = [periodic_correction(na_X1+nona_X1)]
                else:
                    chi_of_interest = [periodic_correction(nona_X1+prot_X1)]


        ##appending merged X1s and X2s to a list
#        print('number of chis', len(chi_of_interest))
        for i in chi_of_interest:
            current_dists.append(i)
            noh2odists.append(i)
            all_chis.append(i)
        
        resstates = [ast.literal_eval(line.rstrip('\n')) for line in open(filename)]

        for i in range(len(resstates)):
            current_states.append(list(np.sort(resstates[i])))
            noh2olimits.append(list(np.sort(resstates[i])))
            all_states.append(list(np.sort(resstates[i])))
        
            
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
        "CALCULATING SINGLE RESIDUE ENTROPY WITHOUT WATER"
        noh2oentropies.append(calculate_entropy(current_states,current_dists))
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
        'ADDING WATER'
        if simulation_pair=='na_nona':
            h2o_dists=[periodic_correction_h2o(nadisth2opsi+nonadisth2opsi),periodic_correction_h2o(nadisth2ophi+nonadisth2ophi)]
        else:
            h2o_dists=[periodic_correction_h2o(nonadisth2opsi+protdisth2opsi),periodic_correction_h2o(nonadisth2ophi+protdisth2ophi)]
        
        current_dists.append(h2o_dists[0])
        current_dists.append(h2o_dists[1])
        
        if simulation_pair=='na_nona':    
            filename='definitive_results/state_intersects/na_nona/'+water_pocket_site+'state_intersects.txt'
#            filename='definitive_results/waterallchis/na_nona_state/'+water_pocket_site+'state_intersects.txt'
        else:
            filename='definitive_results/state_intersects/nona_prot/'+water_pocket_site+'state_intersects.txt'
#            filename='definitive_results/waterallchis/nona_prot_state/'+water_pocket_site+'state_intersects.txt'
        h2o_state_import=[ast.literal_eval(line.rstrip('\n')) for line in open(filename)]
        h2o_states=[]
        for i in range(len(h2o_state_import)):
            h2o_states.append(list(np.sort(h2o_state_import[i])))
            current_states.append(list(np.sort(h2o_state_import[i])))
            
        h2o_entropy=calculate_entropy(h2o_states,h2o_dists)
        
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
        "CALCULATING SINGLE RESIDUE ENTROPY WITH WATER"
        entropy_with_h2o=calculate_entropy(current_states,current_dists)
        entropies.append(entropy_with_h2o)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
    'Appending the water to the total joint vector'
    ##THIS IS APPENDED AFTER EACH RESIDUE IS DEALT WITH
    ##TO AVOID APPENDING IT TWICE
    for i in range(len(h2o_states)):
        all_states.append(list(np.sort(h2o_states[i])))
    all_chis.append(h2o_dists[0])
    all_chis.append(h2o_dists[1])
    
#    print('len all chis', len(all_chis))
#    print('len all states', len(all_states))
    ##this is to double check the match the distributions
#    for i in range(len(all_chis)):
#        plt.figure()
#        plt.hist([item for item in all_chis[i] if item!=10000.0],bins=90)
#        for j in range(len(all_states[i])-1):
#            plt.axvline(all_states[i][j],color='r')
#            
#        plt.show()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
    "CALCULATING JOINT ENTROPY & MUTUAL INFO"
    joint_entropy=calculate_entropy(all_states,all_chis)
    "CALCULATING JOINT ENTROPY & MUTUAL INFO NO WATER"
    noh2ojoint_entropy=calculate_entropy(noh2olimits,noh2odists)
        
    mut_inf_noh2o=sum(noh2oentropies) - noh2ojoint_entropy
    mutual_info_withh2o=sum(entropies)-joint_entropy
    
    ##conditional mutual info = H_xz + H_yz - H_xyz - H_z
    con_mut_inf = mutual_info_withh2o - h2o_entropy
    ii= mut_inf_noh2o - con_mut_inf
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
    "APPENDING THE DATA TO OUTPUT MATRICES"    
    
    interact_matrix[selection1][selection2]=ii
    info_matrix[selection1][selection2]=mut_inf_noh2o
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print('Mu OR')    
    print(simulation_pair)
    print(water_pocket_site)
    print('process', process)
    print('\n')

    print('transfer between residues ', names[selection1][0:-4], 'and ', names[selection2][0:-4])
    print('interaction information =', ii)
    print('Conditional info on h2o =',con_mut_inf)
    print('I('+ names[selection1][0:-4]+','+names[selection2][0:-4]+') = ', mut_inf_noh2o)
    print('H('+names[selection1][0:-4]+') = ', noh2oentropies[0])
    print('H('+names[selection2][0:-4]+') = ', noh2oentropies[1])
    print('H('+names[selection1][0:-4]+','+names[selection2][0:-4]+') = ', noh2ojoint_entropy)
    print(strftime("%Y.%m.%d %H:%M:%S", gmtime()))
    
    print('\n')
    print('##############################################################')
    print('\n')
    
    return [mut_inf_noh2o, ii]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'END OF FUNCTIONS'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## simulation pairs are {na_nona, nona_prot}
print('pairs are na_nona or nona_prot')
simulation_pair=input('choose simulation pair: ',)
print('\n')
print(simulation_pair, 'selected')

## sites are {n150, n749, c647, d349, w450}
print('water pocket sites are n150, n749, c647, d349, or w450')
water_pocket_site=input('choose water pocket site: ')
print('\n')
print(water_pocket_site, 'selected')

# files1 = [f.split("chi1/")[1] for f in glob.glob("definitive_results/naallchis/chi1/" + "**/*.xvg", recursive=True)]
# files2 = [g.split("chi1/")[1] for g in glob.glob("definitive_results/nonaallchis/chi1/" + "**/*.xvg", recursive=True)]
# files3 = [h.split("chi1/")[1] for h in glob.glob("definitive_results/protallchis/chi1/" + "**/*.xvg", recursive=True)]

##looking at just the prolines
# files1 = [f.split("chi1/")[1] for f in glob.glob("definitive_results/naallchis/chi1/" + "**/*.xvg", recursive=True)]
# files2 = [g.split("chi1/")[1] for g in glob.glob("definitive_results/nonaallchis/chi1/" + "**/*.xvg", recursive=True)]
# files3 = [h.split("chi1/")[1] for h in glob.glob("definitive_results/protallchis/chi1/" + "**/*.xvg", recursive=True)]

file_list=[files1,files2,files3]
files_ordered=[[],[],[]]

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

water_list=['n150.txt','n749.txt','c647.txt','w450.txt','d349.txt']
for i in water_list:
    files_ordered[0].insert(1,i)

gro_res_list=range(len(files_ordered[0])-2)

interact_matrix=np.zeros((len(gro_res_list),len(gro_res_list)))
info_matrix=np.zeros((len(gro_res_list),len(gro_res_list)))

inf_mat_indices = list(itertools.combinations(gro_res_list,2))
process='test'

#SEE BELOW FOR PARALLELISATION

#extract_inf_mat_row((5,27))
#cores=46
#cores=23
#checkpoint_number=input('set checkpoint number: ')
#
#for process in range(int(checkpoint_number), 560):
#    print('Starting parallel process ' +str(process))
#    if __name__ == '__main__':
#        ##Pool(x) specifies x cores for use
#       with Pool(cores) as p:
#           ##p.map(function, vector of i values)
#           outputs=p.map(extract_inf_mat_row,inf_mat_indices[process*cores:(process+1)*cores])
#           print(strftime("%Y.%m.%d %H:%M:%S", gmtime()))
#           
#    for i in range(cores):
#   
#           info_matrix[gro_res_list.index(inf_mat_indices[process*cores+i][0])][gro_res_list.index(inf_mat_indices[process*cores+i][1])]=outputs[i][0]
#           interact_matrix[gro_res_list.index(inf_mat_indices[process*cores+i][0])][gro_res_list.index(inf_mat_indices[process*cores+i][1])]=outputs[i][1]
#
#    filename2='definitive_results/heatplots/'+simulation_pair+'_infmat'+str(checkpoint_number)+'.txt'
#    np.savetxt(filename2,np.matrix(info_matrix),fmt='%.4f')
#    
#    filename='definitive_results/heatplots/'+water_pocket_site+'_'+simulation_pair+'_intmat'+str(checkpoint_number)+'.txt'
#    np.savetxt(filename,np.matrix(interact_matrix),fmt='%.4f')   
#    
#    print('finished parallel process '+str(process))
#   
#filename2='definitive_results/heatplots/finished/'+simulation_pair+'_infmat_final'+str(checkpoint_number)+'.txt'
#np.savetxt(filename2,np.matrix(info_matrix),fmt='%.4f')    
#
#filename='definitive_results/heatplots/finished/'+water_pocket_site+'_'+simulation_pair+'_intmat_final'+str(checkpoint_number)+'.txt'
#np.savetxt(filename,np.matrix(interact_matrix),fmt='%.4f')   
