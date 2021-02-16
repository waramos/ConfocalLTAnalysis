# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:37:26 2020

@author: willr
"""

#%%Setup
import os #os API for folder/file management
import pandas as pd #pandas is an API that will help us import the csv files
import numpy as np #an API that will allow us to create arrays and import scientific constants
from numpy import mean
import matplotlib.pyplot as plt #matplot/pyplot API for plotting
from scipy.signal import argrelmax, argrelmin 

#%%Simulation Idea
#what if we were to run a monte carlo simulation to see what a "dendrite" would look like in terms of the centroid offset between puncta if these compartments had no interaction 
#Constants to consider:
#length of dendrite, l
#velocity of compartments/vesicles, v
#rate of vesicle production, r


#%%Analysis Functions
# s = 0.1315 this is the size of a step for dist.
#threshold should be twofold above the background

#Defining Puncta and their Peaks
def puncta(x, y):
    #this function will cut out noise to only include signal that is considered a puncta based on its amplitude being greater than 2*baseline
    list_relmin = argrelmin(y) #creates list containing all relative minima
    y_rm = []
    for pts in list_relmin:
        y_rm.append(y[pts])
    b = mean(y_rm) #averages the local min to approx. the baseline signal level
    t = 2 * b
    xlist = []
    ylist = []
    count = 0
    for i in range(len(x)):
        if y[i]>t:
            xlist.append(x[i])
            ylist.append(y[i])
        else:
            count += 1
    #print(count, "Values have been removed as they were below the threshold of Gray Value =", t)
    return xlist, ylist

def extrema(x, y, method):
#This function will allow us to either find the relative maxima (centroid), or find the minima on either side of peaks from which we can then calculate the width (w) and approximate the centroid (c) by: x + w/2 = c where x is the first rel min, aka the left side of the peak
    if method == "max":
        maxi = argrelmax(np.array(y))[0]
        list_maxi = []
        for m in range(len(maxi)):
            list_maxi.append(int(maxi[m]))
        list_xmax = []
        list_ymax = []
        for k in list_maxi:
            list_xmax.append(x[k])
            list_ymax.append(y[k]) 
        return (list_xmax, list_ymax, maxi) #gives us coordinates as well as the indices
    if method == "min":
        #Gives rel minima
        mini = argrelmin(np.array(y))[0]
        list_mini = []
        for n in range(len(mini)):
            list_mini.append(int(mini[n]))
        list_xmin = []
        list_ymin = []
        for l in list_mini:
            list_xmin.append(x[l])
            list_ymin.append(y[l])
        return (list_xmin, list_ymin, mini) #tuples prevent the data from being altered
    

def defpeak(x, y):
    #Takes the cleaned up data (baseline subtracted)
    #from the rel min, we'll be able to find the widths by counting every other even # one as start and every other even one as an end
    ind = extrema(x, y, method = "min")[2] #gives us the indices that are rel min.
    peak_values = [] #*puncta values, ie all indices that describe a single puncta
    list_puncta = [] #contains all peak value lists
    
    m1_centroids = extrema(x, y, method = "max") #gives rel max indices
    xm1, ym1 = m1_centroids[0], m1_centroids[1] #Gives us the exact values for the rel max.
    
    #Lists containing all data points that comprise a puncta will be appended to a larger list
    #will allow us to measure their width as well as find the centroid as the midpoint of the punta rather than as the peak or rel max.
    for n in range(len(ind)):
        if n == 0:
            peak_values.append(int(ind[n]))
        elif n!=0 and n%2 == 0:
            #index of element in list is even
            peak_values.append(int(ind[n])) #appended value is the index value where the peak starts
        elif n%2 == 1:
            peak_values.append(int(ind[n])) #appended value is the index value where the peak ends
            vr = np.arange(peak_values[0], peak_values[1], 1)
            list_puncta.append(vr.tolist())
            #gives us all indices that will comprise the puncta as a list that is then appended to a larger list that will contain all of these puncta indice lists
            peak_values = [] #resets the indice list
    
    sum_len = 0 #sum of all lengths
    for p in range(len(list_puncta)): #p is the number of puncta in the list
        sum_len += len(list_puncta[p])
    avgw = (sum_len/len(list_puncta)) * 0.1315
    
    #puncta = list with values defined as part of the puncta
    #RETURN: centroid x, centroid y, list of puncta, average width
    return xm1, ym1, list_puncta, avgw #avga = average peak amplitude?


#example of structuring of data below
#L1 = [[1, 2, 3, 4, 5, 6, 7, 8] ,....]
#L2 = [[1, 2, 3], [7, 8, 9] ,....]
#L1[0][n]

def compare(L1, L2, xm1, xm2, x1, x2):
    #note that L1, L2 are the puncta lists while xm1 and xm2 are the actual centroids
    #any given index [i] should correspond to puncta, p_i, with centroid xm_i
    #the values within p_i correspond to the indices within the original list (x1, y1 if L1 for example) that make up said puncta
    #1, 2 of the simply denote the channel that the values correspond to
    list_offset = []
    for i1 in range(len(L1)):
        for i2 in range(len(L2)):
            p_n = [] #puncta values
            p2_n = [] #puncta Ch2 values --> to be updated as we run thru all Ch2 puncta & check to see if they fit our conditions in order for offset to be considered between xm1_n of p_n and xm2_n of p2_n
            #print("this is i1", i1, "this is i2", i2)
            for n in range(len(L1[i1])):
                p_n.append(x1[L1[i1][n]]) #puncta 1 (p_n) defined (x coordinate value)
            for n in range(len(L2[i2])):
                p2_n.append(x2[L2[i2][n]]) #puncta 2 (p2_n) defined (x coordinate value)
            #print("this is pn", p_n, "this is p2", p2_n)

            #Condition 1: minimum overlap of 2 data points
            matchcount = 0
            #perhaps i should instead consider that the x values dont exactly have to overlap and that instead the edges of puncta could be a certain distance away from each other (ie, the difference between at least two values has to be less than a micron (given that the average puncta width is no more than...say...a micron? --> then this could perhaps indeed be a valid way of setting the conditional !))
            gap = []
            for j1 in range(len(p_n)): #scans through all Ch1 puncta values
                for j2 in range(len(p2_n)): #scans thru all Ch2 puncta values
                    gap.append(abs(p_n[j1] - p2_n[j2]))
                    if p_n[j1] == p2_n[j2]:
                        matchcount += 1
            if min(gap)<0.263 or matchcount<2:
                #Condition 2: the offset cannot be more than 2 micron
                off = abs(xm1[i1] - xm2[i2])
                #if negative, its to the right, if +, its to the left
                if -2 < off < 2:
                    list_offset.append(abs(off))
    return list_offset


def compare2(xm1, xm2):
    #note that L1, L2 are the puncta lists while xm1 and xm2 are the actual centroids
    #any given index [i] should correspond to puncta, p_i, with centroid xm_i
    #the values within p_i correspond to the indices within the original list (x1, y1 if L1 for example) that make up said puncta
    #1, 2 of the simply denote the channel that the values correspond to
    list_offset = []
    for i1 in range(len(xm1)):
        for i2 in range(len(xm2)):
    #Condition 2: the offset cannot be more than 2 micron
            off = abs(xm1[i1] - xm2[i2])
            #if negative, its to the right, if +, its to the left
            if -2< off < 2:
                list_offset.append(off)
                    
    return list_offset


#%%Bicucculine Data Analysis
#this function will take the previous functions of comparison and then apply them to conduct the analysis of the line trace data
def ANALYSIS(path, condition):
    os.chdir(path) #Set folder that we'll have the data in, for ease, it's either path_bic or path_con 
    list_offset = []
    
    for i in range(9):
        for j in condition:
            #reading in files
            if "Bicucculine" in path:
                filename = "Bicucculine Field " + str(i+1) + "-" + j +".csv" #dynamic file name allows us to go through all files
            elif "Untreated" in path:
                filename = "Untreated Field " + str(i+1) + "-" + j +".csv"
            dataFile = pd.read_csv(path + "/" + filename, sep = ",")
            x_dis = dataFile.Distance
            xraw = np.array(x_dis.values.tolist()) #x axis: distance
            
            y_gv = dataFile.Gray_Value 
            yraw = np.array(y_gv.values.tolist()) #y axis: gray values
            if j =="ST3":
                x1, y1 = puncta(xraw, yraw)
                xm1, ym1, L1, avgw1 = defpeak(x1, y1)
                
            else:
                x2, y2, = puncta(xraw, yraw)
                xm2, ym2, L2, avgw2 = defpeak(x2, y2)
        
        list_off = compare(L1, L2, xm1, xm2, x1, x2)
        #list_off = compare2(xm1, xm2)
        list_offset.extend(list_off)
    
    #Lists that will be averaged to produce the graphs
    return list_offset


#%%Information on Experiments

#Drive might change depending on computer
drive = "E"

#ST3 - EEA1 experiments
path_bic = drive + ":/Green Lab/2020 Data Analysis/Ok/ST3-EEA1 CSV files/Bicucculine"
path_con = drive + ":/Green Lab/2020 Data Analysis/Ok/ST3-EEA1 CSV files/Untreated"

#ST3 - WGA experiments
path_bicW = drive + ":/Green Lab/2020 Data Analysis/Ok/ST3-WGA CSV Files/Bicucculine"
path_conW = drive + ":/Green Lab/2020 Data Analysis/Ok/ST3-WGA CSV Files/Untreated"

#Conditions within each experiment to be analyzed
cond = ["EEA1", "ST3"] #initial set of data
cond2 = ["ST3", "WGA"] #second set from WGA Experiments

#Remember that x is in microns and y is in gray value

#%%Graphical Output + Results

#Offset Graphs
#ST3 and EEA1 w/ Bic
data_bic = ANALYSIS(path_bic, cond)

#ST3 and EEA1 Untreated (control)
data_con = ANALYSIS(path_con, cond)

#Offset graphs
plt.hist(data_bic)
plt.legend()
plt.title("Offset Between ST3 and EEA1 Centroids")
plt.xlabel("Offset (" + r'$\mu$' + "m)")
plt.ylabel("Number of EEA1 Puncta")
#plt.xlabel("Centroid Determination Method")
#plt.legend(["Bicucculine", "Untreated"])


#%%
def hplot(path, condition):
    data = ANALYSIS(path, condition)
    if "Bicucculine" in path:
        name = "Bicucculine Field"
    elif "Untreated" in path:
        name = "Untreated Field"
    plt.hist(data)
    plt.title("Offset Between " + condition[0] + " and " + condition[1] + " Centroids in " + name)
    plt.xlabel("Offset (" + r'$\mu$' + "m)")
    plt.ylabel("Number of Puncta")

hplot(path_bic, cond)
hplot(path_con, cond)
hplot(path_bicW, cond2)
hplot(path_conW, cond2)



#%%
#2nd set of experiments
#ST3 and WGA w/ Bic
data_bicW = ANALYSIS(path_bicW, condition = cond2)

#ST3 and WGA Untreated (control)
data_conW = ANALYSIS(path_conW, cond2)

#Offset Graphs


#Overlap Graphs
ol = data_bic[0], data_con[0]
ol_se = data_bic[1], data_con[1]

plt.bar(x = ["Bicucculine"], height = ol[0], yerr = ol_se[0])
plt.bar(x = ["Untreated"], height = ol[1], yerr = ol_se[1])

plt.title("Overlap Between ST3 and WGA Puncta")
plt.ylabel("Overlap (% of ST3 Puncta Width)")



#%%

#Offset Graphs
#ST3 and EEA1 w/ Bic
data_bic = ANALYSIS(path_bic, cond)
h_b = data_bic[2] #, data_bic[4]
er_b = data_bic[3] #, data_bic[5]
#ST3 and EEA1 Untreated (control)
data_con = ANALYSIS(path_con, cond)
h_con = data_con[2] #, data_con[4]
er_con = data_con[3] #, data_con[5]

#Offset graphs
plt.bar(x = ["Bicucculine"], height = h_b, yerr = er_b)
plt.bar(x = ["Untreated"], height = h_con, yerr = er_con)
plt.title("Offset Between ST3 and EEA1 Centroids")
plt.ylabel("Offset (" + r'$\mu$' + "m)")
#plt.xlabel("Centroid Determination Method")
#plt.legend(["Bicucculine", "Untreated"])

#Overlap Graphs
ol = data_bic[0], data_con[0]
ol_se = data_bic[1], data_con[1]

plt.bar(x = ["Bicucculine"], height = ol[0], yerr = ol_se[0])
plt.bar(x = ["Untreated"], height = ol[1], yerr = ol_se[1])
plt.title("Overlap Between ST3 and EEA1 Puncta")
plt.ylabel("Overlap (% of ST3 Puncta Width)")



#2nd set of experiments
#ST3 and WGA w/ Bic
data_bicW = ANALYSIS(path_bicW, cond2)
h_b = data_bicW[2] #, data_bicW[4]
er_b = data_bicW[3] #, data_bicW[5]

#ST3 and WGA Untreated (control)
data_conW = ANALYSIS(path_conW, cond2)
h_con = data_conW[2] #, data_conW[4]
er_con = data_conW[3] #, data_conW[5]

#Offset Graphs
plt.bar(x = ["Bicucculine"], height = h_b, yerr = er_b)
plt.bar(x = ["Untreated"], height = h_con, yerr = er_con)
plt.title("Offset Between ST3 and WGA Centroids")
plt.ylabel("Offset (" + r'$\mu$' + "m)")
#plt.xlabel("Centroid Determination Method")
#plt.legend(["Bicucculine", "Untreated"])

#Overlap Graphs
ol = data_bic[0], data_con[0]
ol_se = data_bic[1], data_con[1]

plt.bar(x = ["Bicucculine"], height = ol[0], yerr = ol_se[0])
plt.bar(x = ["Untreated"], height = ol[1], yerr = ol_se[1])

plt.title("Overlap Between ST3 and WGA Puncta")
plt.ylabel("Overlap (% of ST3 Puncta Width)")


#%%Better Graphs

#%%Line trace graphs


#%%Need to make offset calculations conditional
