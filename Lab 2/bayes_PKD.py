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

def printCounter(counter):
    for first in counter.keys():
        print(f'{first}', end='')
        if first == '': print('\'\'', end='')
        print(':')
        for second in counter[first].keys():
            if second == '':
                print('    %-18s %s' %('\'\'', counter[first][second]))
            else:
                print('    %-18s %s' %(second, counter[first][second]))
        print()
    return

def probRaceGivenUnarmed():
    results = countRaceAndUnarmed(INCIDENT_LIST)
    prob = dict()
    for status in results.keys():
        for race in results[status].keys():
            if status not in prob: prob[status] = dict()
            totalcounter = [results[status][race] for race in results[status]]
            prob[status][race] = results[status][race] / sum(totalcounter)
    return prob

def raceQueries():
    return ("White", "Black", "Hispanic", "Asian")

def statusQueries():
    return ("Allegedly Armed", "Unarmed", "Unclear")

def getQuery():
    query_list = []
    for race in raceQueries():
        for status in statusQueries():
            query_list.append((race, status))
    return query_list

def printProb():
    prob = probRaceGivenUnarmed()
    queries = getQuery()

    def makeProbString(race, status):
        statusBool = {
            'Allegedly Armed':'true',
            'Unarmed':'false',
            'Unclear':'unclear'
        }
        parts = ['p(race=',
                 race.lower(),
                 ' | person_was_killed_by_police=true, ', 
                 'victim_was_armed=',
                 statusBool[status],
                 ')'
        ]
        return ''.join(parts)

    for race, status in queries:
        prob_string = makeProbString(race, status)
        print('%-80s %s' %(prob_string, prob[status][race]))
    return

def printSpecificProb():
    """
    Assumes query races are only White, Black, Hispanic, Asian
    """
    def printSpecificProb(status):
        for race in raceQueries():
            probRaceStatus = probRaceGivenUnarmed()
            summands = [
                0.8 * probRaceStatus[status][race],
                0.2 * probRaceStatus['Unclear'][race]
            ]
            prob = sum(summands)
            print('%-30s %s' %(f'p(race={race.lower()} | pwkby=true)', prob))
        print('\n')

    print('Case 1') # where {"prob_armed":0.8, "prob_unclear":0.2}
    printSpecificProb('Allegedly Armed')
    
    print('Case 2') # where {"prob_unarmed":0.8, "prob_unclear":0.2}
    printSpecificProb('Unarmed')
    




if __name__=='__main__':
    INCIDENT_LIST = open_PKDv2.readPKD('assets/police_killings.csv')
    INCIDENT_TOTAL = len(INCIDENT_LIST)

    ### Part (a) ###
    # U.S. Census race data: https://www.census.gov/library/visualizations/interactive/race-and-ethnicity-in-the-united-state-2010-and-2020-census.html
    P_WHITE = 0.616     # Group: White alone
    P_ASIAN = 0.06      # Group: Asian alone
    P_BLACK = 0.124     # Group: Black or African American alone
    P_HISPANIC = 0.187  # Group: Hispanic or Latino
    TOTAL_P_RACE = P_WHITE + P_ASIAN + P_BLACK + P_HISPANIC # = 0.987

    ### Part (b) ###
    #printProb()

    ### Part (c) ###
    #printSpecificProb()

    printCounter( countRaceAndUnarmed(INCIDENT_LIST))