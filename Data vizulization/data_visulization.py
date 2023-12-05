import pandas as pd
from dateutil import relativedelta
from datetime import datetime
import math
import operator
import matplotlib.pyplot as plt

data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")
data2 = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx", "Sheet2")
n = len(data)
n2 = len(data2)

############### Calculate average age at first visitation ###############
days = 0
average_years = 0
age_array = [0] * 100
indexes_to_remove = []

for i in range(n):
        
    date_birth = datetime.strptime(data['birth'][i], "%d-%b-%y")
    date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
    difference = relativedelta.relativedelta(date_visitation, date_birth)
    
    if difference.years < 0 or difference.years > 18:
        indexes_to_remove.append(i)
        print("OBS old/misregistred: index", i, ", year difference:", difference.years)
    else:
        age_array[difference.years] = operator.add(age_array[difference.years], 1)

    days = days + (difference.years * 365.25 + difference.months * (365.25 / 12) + difference.days)
    average_years = days/365.25/n
    
print("Indexes to remove: ", indexes_to_remove)
print("Average age at first visitation:", average_years, " years")      

total_years_int = int(average_years)

total_months = ((average_years - int(average_years)) * 12)
total_months_int = int(total_months)

total_days = ((total_months - total_months_int) * (365.25/12))
total_days_int = int(total_days)

print("Average age at first visitation:", total_years_int, " years,", total_months_int, "months, and ", total_days_int, "days")

data = data.drop(indexes_to_remove).reset_index(drop=True)

############### Plotting age distribution as bar chart ###############
fig = plt.figure(figsize=(10, 6))
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
plt.bar(labels, age_array[:len(labels)], color='navy', width=0.4)
plt.xlabel("Age", fontsize=13)
plt.ylabel("Number of patients", fontsize=13)
plt.yticks(fontsize=13)
plt.title("Patient age distribution at first clinical examination", fontsize=15)
plt.show()

print("Age distribution at first visitation:", age_array[:18])
#print("The rest:", age_array[18:])

############### Calculate gender distribution ###############
girls = 0
boys = 0

for i in range(len(data)):
    
    gender = data['sex'][i]
    
    if gender == 0:
        boys += 1
    elif gender == 1:
        girls += 1
        
print("Number of girls:", girls, " |  Number of boys:", boys)

############### Calculate arthritis type distribution ###############
type_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(data)):
    
    arthritis_type = data['type'][i]
    
    if math.isnan(arthritis_type):
        type_array[10] = operator.add(type_array[10], 1)
    else:
        arthritis_type = int(arthritis_type)
        type_array[arthritis_type] = operator.add(type_array[arthritis_type], 1)

print("Distribution of arthritis type of patients:", type_array)

############### Plotting distribution of JIA types as bar chart ###############
fig = plt.figure(figsize=(10, 12))
labels = ['pauci', 'poly', 'systemic', 'enthecitis', 'psoriasis related', 'poly+psoriasis', 'systemic+psoriasis', 'pauci+psoriasis', 'CRMO', 'CRMO+JIA', 'Not registered']
plt.bar(labels, type_array, color='navy', width=0.4)
plt.ylabel("Number of patients", fontsize=17)
plt.title("Subtype age distribution at first clinical examination", fontsize=20)
plt.xticks(rotation='vertical')
plt.xticks(rotation=45)
plt.yticks(fontsize=17)
plt.show()

############### Vizualize overall TMJ involvment status ###############
status_array = [0, 0, 0, 0, 0]

for i in range(len(data)):
    
    TMJ_status = data['overall TMJ involvement'][i]
    
    if math.isnan(TMJ_status):
        status_array[0] = operator.add(status_array[0], 1)
    elif int(TMJ_status) == 0 or int(TMJ_status) == 1 or int(TMJ_status) == 2 or int(TMJ_status) == 3:
        status_array[int(TMJ_status)] = operator.add(status_array[int(TMJ_status)], 1)
    else:
        status_array[4] = operator.add(status_array[4], 1)

# Status code: [0, 1, 2, 3]
print("Overall TMJ status of patients:", status_array)

############### Plotting overall TMJ involvement status as bar chart ###############
fig = plt.figure(figsize=(10, 10))
labels = ['No TMJ', 'Right TMJ', 'Left TMJ', 'Both TMJ', "Misregistered"]
plt.bar(labels, status_array, color='navy', width=0.4)
plt.xticks(rotation='vertical')
plt.ylabel("Number of patients", fontsize=17)
plt.title("Overall TMJ involvement status distribution", fontsize=20)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=17)
plt.show()