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

def makeRaceCounter(init_value):
    races = printRaces(suppress=True)
    return makeCounter(races, init_value)

def makeRaceUnarmedCounter():
    unarmed = printUnarmedLabels(suppress=True)
    raceCounter = makeRaceCounter(0)
    return makeCounter(unarmed, raceCounter)

def makeRaceAgeCounter():
    ages = ['<20', '>=20']
    age_counter = makeCounter(ages, 0)
    return makeRaceCounter(age_counter)

def countRaceAndUnarmed():
    counter = makeRaceUnarmedCounter()
    for Inc in INCIDENT_LIST:
        counter[Inc.unarmed][Inc.victims_race] += 1
    return counter

def countRaceAndAge():
    counter = makeRaceAgeCounter()
    for Inc in INCIDENT_LIST:
        # filter out '40s' or 'Unknown
        if not Inc.victims_age.isnumeric(): continue
        if int(Inc.victims_age) < 20:
            counter[Inc.victims_race]['<20'] += 1
        else:
            counter[Inc.victims_race]['>=20'] += 1
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

def probFirstGivenSecond(counter):
    prob = dict()
    for first in counter.keys():
        for second in counter[first].keys():
            if first not in prob: prob[first] = dict()
            totalcounter = [counter[first][second] for second in counter[first]]
            if sum(totalcounter) == 0: prob[first][second] = 0
            else:
                prob[first][second] = counter[first][second] / sum(totalcounter)
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

def printPartBProb():
    prob = probFirstGivenSecond(
        countRaceAndUnarmed()
    )
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

def printSpecificProb(status):
    probRaceGivenKilled = dict()
    for race in raceQueries():
        probRaceStatus = probFirstGivenSecond(
            countRaceAndUnarmed()
        )
        summands = [
            0.8 * probRaceStatus[status][race],
            0.2 * probRaceStatus['Unclear'][race]
        ]
        prob = sum(summands)
        print('%-30s %s' %(f'p(race={race.lower()} | pwkby=true)', prob))
        probRaceGivenKilled[race] = prob
    return probRaceGivenKilled

def printPartCProb():
    """
    Assumes query races are only White, Black, Hispanic, Asian
    """
    print('Case 1') # where {"prob_armed":0.8, "prob_unclear":0.2}
    printSpecificProb('Allegedly Armed')
    
    print('\nCase 2') # where {"prob_unarmed":0.8, "prob_unclear":0.2}
    printSpecificProb('Unarmed')

def bayesCorrect():
    prob = probFirstGivenSecond(
            countRaceAndAge()
    )

def predictionStep():
    bel_white = p_white_unarmed*p
    return [bel_white, bel_black, bel_hispanic, bel_asian]

def printPartDProb():
    prob = probFirstGivenSecond(
            countRaceAndAge()
    )
    for race in raceQueries():
        print('%-60s %s' %(f'p(age<20 | person_was_killed_by_police=true, race={race.lower()})',
            prob[race]['<20']))



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
    printPartBProb()

    ### Part (c) ###
    printPartCProb()

    ### 
    printPartDProb()