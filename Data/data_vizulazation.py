import pandas as pd
from dateutil import relativedelta
from datetime import datetime
import math
import operator
import matplotlib.pyplot as plt
import numpy as np     

# Import data
data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")
data2 = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet2")
n = len(data)
n2 = len(data2)

# Calculate average age at first visitation (average_age_first_visitation)
days = 0
age_array = [0] * 100
relativedeltas = []

for i in range(n):
        
    # Parse the dates from strings into datetime objects
    date_birth = datetime.strptime(data['birth'][i], "%d-%b-%y")
    
    # Convert first visitation value to datetime
    date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
   
    # Calculate the difference between the two dates
    difference = relativedelta.relativedelta(date_visitation, date_birth)
    
    if difference.years > 0 and difference.years < 19:
        
        if difference.years < 0:
            difference.years = 100 + difference.years
            print("OBS old/misregistred: index", i, ", year difference:", difference.years)
               
        if difference.years < 100:
            age_array[difference.years] = operator.add(age_array[difference.years], 1)    
            
        days = days + (difference.years * 365.25 + difference.months * (365.25/12) + difference.days)
    
average_years = days/365.25/n
    
print("Average age at first visitation:", average_years, " years")      

total_years_int = int(average_years)

total_months = ((average_years - int(average_years)) * 12)
total_months_int = int(total_months)

total_days = ((total_months - total_months_int) * (365.25/12))
total_days_int = int(total_days)

print("Average age at first visitation:", total_years_int, " years,", total_months_int, "months, and ", total_days_int, "days") 


# Plotting age distribution as bar chart
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Number of patients')
ax.set_xlabel('Age')
labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
ax.bar(labels, age_array[:len(labels)])
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Number of patients')
ax.set_xlabel('Age')
plt.plot(age_array)
plt.show()

print("Age distribution at first visitation:", age_array[:18])
#print("The rest:", age_array[18:])

# Calculate gender distribution
girls = 0
boys = 0

for i in range(n):
    
    gender = data['sex'][i]
    
    if gender == 0:
        boys += 1
    elif gender == 1:
        girls += 1
        
print("Number of girls:", girls, " |  Number of boys:", boys)

# Calculate arthritis type distribution
type_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(n):
    
    arthritis_type = data['type'][i]
    
    if math.isnan(arthritis_type):
        type_array[10] = operator.add(type_array[10], 1)
    else:
        arthritis_type = int(arthritis_type)
        type_array[arthritis_type] = operator.add(type_array[arthritis_type], 1)

# [pauci, poly, systemic, enthecitis, psoriasis related, poly+psoriasis, systemic+psoriasis, pauci+psoriasis, CRMO, CRMO+JIA, Not registered]
print("Distribution of arthritis type of patients:", type_array)
        
# Plotting distribution of JIA types as bar chart
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Number of patients')
labels = ['pauci','poly','systemic','enthecitis','psoriasis related','poly+psoriasis','systemic+psoriasis','pauci+psoriasis', 'CRMO','CRMO+JIA','Not registered']
plt.xticks(rotation='vertical')
ax.bar(labels, type_array)
plt.show()

# Vizualize overall TMJ involvment status
status_array = [0, 0, 0, 0, 0]

for i in range(n):
    
    TMJ_status = data['overall TMJ involvement'][i]
    
    if math.isnan(TMJ_status):
        status_array[0] = operator.add(status_array[0], 1)
    elif int(TMJ_status) == 0 or int(TMJ_status) == 1 or int(TMJ_status) == 2 or int(TMJ_status) == 3:
        status_array[int(TMJ_status)] = operator.add(status_array[int(TMJ_status)], 1)
    else:
        status_array[4] = operator.add(status_array[4], 1)

# Status code: [0, 1, 2, 3, 4, 5, 6, 7]
print("Overall TMJ status of patients:", status_array)

# Plotting overall TMJ involvement status as bar chart
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Number of patients')
labels = ['No TMJ involvment','Right TMJ','Left TMJ','Both TMJ', "Misregistered"]
plt.xticks(rotation='vertical')
ax.bar(labels, status_array)
plt.show()

# Import data
data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")
data2 = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet2")
n = len(data)
n2 = len(data2)

# Trajectory
status_matrix = np.zeros(18)

for i in range(n2):
    
    status = data2['overall TMJ involvement'][i]
    
    if status == 1 or status == 2 or status == 3:
        
        row = data2.loc[i,:]
        status_matrix = np.vstack([status_matrix, row])
    
# Remove first row
status_matrix = np.delete(status_matrix, 0, 0)

# Remove first column
status_matrix = np.delete(status_matrix, 0, 1)

# Plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Involvment status')
ax.set_xlabel('Visitation number')
plt.xlim([0, 17])
#for i in range(len(status_matrix)):
for i in range(40,41):
    plt.plot(status_matrix[i])
plt.show()

