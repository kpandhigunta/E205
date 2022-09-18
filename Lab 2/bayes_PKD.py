"""
Author: Sean Wu
Email: sywu@g.hmc.edu
Date of Creation: Sep 17, 2022
"""

import open_PKD # Access Police Killing Dataset, modified from Lab 1 

def printData(param, suppress=False):
    data = set()
    for inc in INCIDENT_LIST:
        data.add(eval(('inc.'+param)))
    if not suppress:
        print(param, '->', data)
    return data

def printUnarmedLabels(suppress=False):
    return printData('unarmed', suppress)

def printRaces(suppress=False):
    return printData('victims_race', suppress)

def makeRaceCounter():
    races = printRaces(suppress=True)
    


def countUnarmed(incident_list):
    
    unarmed_counter = {
        '':0,
        'Vehicle':0,
        'Allegedly Armed':0,
        'Unclear':0,
        'Unarmed':0
    }
    for inc in incident_list:
        label = inc.unarmed
        unarmed_counter[label] += 1
    return unarmed_counter

def predictVictim(race, was_armed):
    """
    Calculates p(race=race | person_was_killed_by_police=true, victim_was_armed=was_armed)
    
    race is a string, see printRaces()
    was_armed is a string: true, false, or unclear
    """

    return

if __name__=='__main__':
    INCIDENT_LIST = open_PKD.readPKD('assets/police_killings.csv')
    INCIDENT_TOTAL = len(INCIDENT_LIST)

    # Part (a)
    # U.S. Census race data: https://www.census.gov/library/visualizations/interactive/race-and-ethnicity-in-the-united-state-2010-and-2020-census.html
    P_WHITE = 0.616     # Group: White alone
    P_ASIAN = 0.06      # Group: Asian alone
    P_BLACK = 0.124     # Group: Black or African American alone
    P_HISPANIC = 0.187  # Group: Hispanic or Latino
    TOTAL_P_RACE = P_WHITE + P_ASIAN + P_BLACK + P_HISPANIC # = 0.987

    printRaces()
    printUnarmedLabels()
    



