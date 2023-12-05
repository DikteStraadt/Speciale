# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:00:27 2023

@author: lenat
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator

data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")
n = len(data)
indexes_to_remove = [0, 4, 146, 150, 178, 269, 297, 421, 532, 557, 558, 621, 646, 654, 685, 852, 901, 917, 971]
data = data.drop(indexes_to_remove).reset_index(drop=True)

xls = pd.ExcelFile('C:/Users/User/Downloads/Master_Excel_Sep4.xlsx')
df2 = pd.read_excel(xls, 'Sheet2')

# Loop henaf rækken - tjek om cell er empty, hvis ikke incrementer counter
visit_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
visnum = 0
df = pd.DataFrame(df2)
#df.remove('overall TMJ involvement')
newdf = df.drop("overall TMJ involvement", axis='columns')
# Når der rammes en empty cell - lig ind i array
for index, row in newdf.iterrows():
    #print(row['involvement_status_0'])
    for columnIndex, value in row.items():
        if np.isnan(value):
            pass
        else:
            visnum = newdf.columns.get_loc(columnIndex)
            #print(newdf.columns.get_loc(columnIndex))
    
    #print(visnum)
    visit_array[visnum] = operator.add(visit_array[visnum], 1)

fig = plt.figure(figsize=(10, 6))
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
plt.bar(labels, visit_array, color='navy', width=0.4)
plt.title("Number of patients with a set number of examinations", fontsize=15)
plt.xlabel('Number of examinations', fontsize=13)
plt.ylabel("Number of patients", fontsize=13)
plt.yticks(fontsize=17)
plt.show()

# Import data
data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet2")
n = len(data)

data2 = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet2")
data2.pop('overall TMJ involvement')
n2 = len(data2)

# Trajectory
status_matrix = np.zeros(18)

for i in range(n):
    
    status = data['overall TMJ involvement'][i]
    
    row = data.loc[i,:]
    status_matrix = np.vstack([status_matrix, row])


# Remove first row
status_matrix = np.delete(status_matrix, 0, 0)

# Remove first column
status_matrix = np.delete(status_matrix, 0, 1)


overview_array = [0, 0, 0, 0, 0, 0, 0, 0]


for row in status_matrix:
  for item in row:
    if item == 0:
        overview_array[0]+=1
        
    if item == 1:
        overview_array[1]+=1
        
    if item == 2:
        overview_array[2]+=1
        
    if item == 3:
        overview_array[3]+=1
        
    if item == 4:
        overview_array[4]+=1
        
    if item == 5:
        overview_array[5]+=1
        
    if item == 6:
        overview_array[6]+=1
        
    if item == 7:
        overview_array[7]+=1

fig = plt.figure(figsize=(10, 12))
labels = ['No TMJ involement', 'TMJ arthritis right', 'TMJ arthritis left', 'TMJ arthritis both', 'TMJ chronic right', 'TMJ chronic left', 'TMJ chronic both' ,'Obs arthritis']
plt.xlabel('Type', fontsize=13)
plt.ylabel('Number of visitations', fontsize=13)
plt.title("TMJ involvement status for all examinations", fontsize=20)
plt.xticks(rotation='vertical')
plt.xticks(rotation=45)
plt.yticks(fontsize=15)
plt.bar(labels, overview_array, color='navy', width=0.4)
plt.show()



