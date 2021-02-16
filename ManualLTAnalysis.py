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
from pandas import DataFrame as DF

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



#%%Bicucculine Data Analysis
#this function will take the previous functions of comparison and then apply them to conduct the analysis of the line trace data
def ANALYSIS(path, condition):
    os.chdir(path) #Set folder that we'll have the data in, for ease, it's either path_bic or path_con 
    
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
                #Saving the data as a .csv
        df = DF([x1, y1, xm1, ym1, x2, y2, xm2, ym2], index = ["x1", "y1", "xc1", "yc1", "x2", "y2", "xc2", "yc2"])
                
            d = df.to_csv(path_or_buf = path, sep = ",")
            return d

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


#%%

#def datatocsv(path, cond):
path = path_con
condition = cond
os.chdir(path)
for i in range(9):
    for j in condition:
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
            df = DF([x1, y1, xm1, ym1], index = ["x1", "y1", "xc1", "yc1"])
                        
            df.to_csv("CLEAN" + filename)
                        
        else:
            x2, y2 = puncta(xraw, yraw)
            xm2, ym2, L2, avgw2 = defpeak(x2, y2)
            df = DF([x2, y2, xm2, ym2], index = ["x2", "y2", "xc2", "yc2"])
                        
            df.to_csv("CLEAN" + filename)
#%%


# #%%
# def hplot(path, condition):
#     data = ANALYSIS(path, condition)
#     if "Bicucculine" in path:
#         name = "Bicucculine Field"
#     elif "Untreated" in path:
#         name = "Untreated Field"
#     plt.hist(data)
#     plt.title("Offset Between " + condition[0] + " and " + condition[1] + " Centroids in " + name)
#     plt.xlabel("Offset (" + r'$\mu$' + "m)")
#     plt.ylabel("Number of Puncta")

# hplot(path_bic, cond)
# hplot(path_con, cond)
# hplot(path_bicW, cond2)
# hplot(path_conW, cond2)

path = path_bic
condition = cond
os.chdir(path)
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
                #Saving the data as a .csv
    df = DF([x1, y1, xm1, ym1, x2, y2, xm2, ym2], index = ["x1", "y1", "xc1", "yc1", "x2", "y2", "xc2", "yc2"])
    df.to_csv("CLEAN Bic Field ST3 and EEA1 " + str(i+1) + ".csv")