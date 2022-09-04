import csv
import numpy as np

class Incident(object):

    def __init__(self, row):    
        self.Victims_name = row[0]
        self.victims_age = row[1]
        self.victims_gender = row[2]
        self.victims_race = row[3]
        self.url = row[4]
        self.date = row[5]
        self.street_address = row[6]
        self.city = row[7]
        self.state = row[8]
        self.zipcode = row[9]
        self.county = row[10]
        self.agency = row[11]
        self.cause_of_death = row[12]
        self.description = row[13]
        self.official_disposition = row[14]
        self.criminal_charges = row[15]
        self.news_article = row[16]
        self.symptoms_of_mental_illness = row[17]
        self.unarmed = row[18]
        self.alleged_weapon = row[19]
        self.alleged_threat_level = row[20]
        self.fleeing = row[21]
        self.body_camera = row[22]
        self.wapo_id = row[23]
        self.off_duty_killing = row[24]
        self.geography = row[25]
        self.id = row[26]
        

# Read in the data
print('1. Opening file.')
with open('police_killings.csv', encoding='utf8') as csvfile:
    print('2. Loading data.')
    readCSV = csv.reader(csvfile, delimiter=',')
    i=0
    incident_list = []
    for row in readCSV:
        incident_list.append(Incident(row))      
    print('3. Done loading data.')


# Sample code that accesses data
race_index = {'black':0,
            'white':1,
            'hispanic':2,
            'asian':3,
            'pacific islander':4,
            'native american':5,
            'unknown race':6}
arr = np.zeros((len(race_index),5), dtype=int)
for i in incident_list[1:7702]:
    age = i.victims_age
    if not age.isnumeric():
        continue
    age = int(age)
    race = i.victims_race.lower()
    if age in range(0, 16):
        arr[race_index[race], 0] += 1
    elif age in range(16, 21):
        arr[race_index[race], 1] += 1
    elif age in range(21, 26):
        arr[race_index[race], 2] += 1
    elif age in range(26, 31):
        arr[race_index[race], 3] += 1
    else:
        arr[race_index[race], 4] += 1
print(arr)
SUM = np.sum(arr)
print('Total', SUM, '\tomitted', 7701 - SUM)

def sum_race(arr, race):
    return np.sum(arr[race_index[race],:])

def prob_age(arr, age_bucket):
    return np.sum(arr[:, age_bucket]) / SUM

prob_arr = np.zeros_like(arr)
prob_arr = ((arr)/ np.sum(arr, axis=0)).T

# def p_age_given_race(arr):
#     for race in race_index:
#         prob_arr[race_index[race], 0] = arr[race_index[race], 0] / sum_race(arr, race)
# p_age_given_race(arr)
print(prob_arr)


        
    
    
    
    