"""
Author: Sean Wu
Email: sywu@g.hmc.edu
Date of Creation: Sep 17, 2022
"""

import open_PKDv2 # Access Police Killing Dataset, modified from Lab 1 

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

def makeCounter(data_list, init_val):
    counter = dict()
    for item in data_list:
        if type(init_val) is not int:
            counter[item] = init_val.copy()
        else:
            counter[item] = init_val
    return counter

def makeRaceCounter():
    races = printRaces(suppress=True)
    return makeCounter(races, 0)

def makeRaceUnarmedCounter():
    unarmed = printUnarmedLabels(suppress=True)
    raceCounter = makeRaceCounter()
    return makeCounter(unarmed, raceCounter)

def countRaceAndUnarmed(incident_list):
    counter = makeRaceUnarmedCounter()
    for Inc in incident_list:
        counter[Inc.unarmed][Inc.victims_race] += 1
    return counter

def printRaceAndUnarmed():
    results = countRaceAndUnarmed(INCIDENT_LIST)
    for armedstatus in results.keys():
        print(f'{armedstatus}', end='')
        if armedstatus == '': print('\'\'', end='')
        print(':')
        for race in results[armedstatus].keys():
            if race == '':
                print('    %-18s %s' %('\'\'', results[armedstatus][race]))
            else:
                print('    %-18s %s' %(race, results[armedstatus][race]))
        print()
    return


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

    printRaceAndUnarmed()