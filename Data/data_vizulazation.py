import pandas as pd
from dateutil import relativedelta
from datetime import datetime
import math
import operator
import matplotlib.pyplot as plt

# Import data
data = pd.read_excel("C:/Users/User/Downloads/Master_Excel_Sep4.xlsx")
n = len(data)

# Calculate average age at first visitation (average_age_first_visitation)
years = 0
months = 0
days = 0

for i in range(n):
        
    # Parse the dates from strings into datetime objects
    date_birth = datetime.strptime(data['birth'][i], "%d-%b-%y")
    
    if math.isnan(data['age_onset'][i]):
       
        # Convert first visitation value to datetime
       date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
       
       # Calculate the difference between the two dates
       difference = relativedelta.relativedelta(date_visitation, date_birth)

       # Print the number of days between the two dates
       # print(difference.years, "years,", difference.months, "months,", difference.days, "days")
       
       years = years + difference.years
       months = months + difference.months
       days = days + difference.days
       
    else:
        
        # Convert age onset value to date       
        year = int(data['age_onset'][i])        
        month = int((data['age_onset'][i] - year) * 12) 

        if month < 1:
            day = int((data['age_onset'][i] - year) * 365.25)
        else:
            day = int((((data['age_onset'][i] - year) * 12) - month) * (365.25/12))
        
        # Print the age on the onset date
        # print("From onset date: i: ", i, ",", year, "years,", month, "months,", day, "days")
        
        years = years + year
        months = months + month
        days = days + day
   
# print("Years: ", years, ", months: ", months, ", days: ", days)  

average_years = years/n
average_months = months/n
average_days = days/n

print("Years: ", average_years, ", months: ", average_months, ", days: ", average_days)  

total_years_int = int(average_years)

total_months = ((average_years - int(average_years)) * 12) + average_months
total_months_int = int(total_months)

total_days = ((total_months - total_months_int) * (365.25/12)) + average_days
total_days_int = int(total_days)

if total_months >= 1:
    total_years_int += 1
    total_months_int -= 12

print("Average age at first visitation:", total_years_int, " years,", total_months_int, "months, and ", total_days_int, "days")      

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
# (https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
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
# (https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['No TMJ involvment','Right TMJ','Left TMJ','Both TMJ', "Misregistered"]
plt.xticks(rotation='vertical')
ax.bar(labels, status_array)
plt.show()
