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
    date_visitation = datetime.strptime(data['first_visitation'][i], "%d-%b-%y")
     
    # Calculate the difference between the two dates
    difference = relativedelta.relativedelta(date_visitation, date_birth)

    # Print the number of days between the two dates
    #print(difference.years, "years,", difference.months, "months,", difference.days, "days")
    
    years = years + difference.years
    months = months + difference.months
    days = days + difference.days
    
    #print("i: ", i, ",", years, "years,", months, "months,", days, "days")
    

print("Average age at first visitation: ", math.floor(years/n), "years,", math.floor(months/n), "months,", math.floor(days/n), "days")

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








# Vizualize overall TMJ involvment status
status_array = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(n):
    
    TMJ_status = data['overall TMJ involvement'][i]
    
    if math.isnan(TMJ_status):
        status_array[0] = operator.add(status_array[0], 1)
    else:
        TMJ_status = int(TMJ_status)
        status_array[TMJ_status] = operator.add(status_array[TMJ_status], 1)

# Status code: [0, 1, 2, 3, 4, 5, 6, 7]
print("Overall TMJ status of patients:", status_array)

# Plotting overall TMJ involvement status as bar chart
# (https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['0','1','2','3','4','5','6','7']
ax.bar(labels, status_array)
plt.show()







# Calculate average number of visitations (average_number_of_visitations)
