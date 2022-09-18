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

def printBanner(label):
    print(
        '\n===========\n',
        ' Part ', label,
        '\n===========\n'
    )
    return

def printPartBProb():
    printBanner('b')
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

def printProbRaceGivenKilled(status, suppress=False):
    prob_race_given_killed = dict()
    for race in raceQueries():
        prob_race_status = probFirstGivenSecond(
            countRaceAndUnarmed()
        )
        summands = [
            0.8 * prob_race_status[status][race],
            0.2 * prob_race_status['Unclear'][race]
        ]
        prob = sum(summands)
        if not suppress:
            print('%-30s %s' %(f'p(race={race.lower()} | pwkby=true)', prob))
        prob_race_given_killed[race] = prob
    return prob_race_given_killed

def printPartCProb():
    """
    Assumes query races are only White, Black, Hispanic, Asian
    """
    printBanner('c')
    
    print('Case 1') # where {"prob_armed":0.8, "prob_unclear":0.2}
    printProbRaceGivenKilled('Allegedly Armed')
    
    print('\nCase 2') # where {"prob_unarmed":0.8, "prob_unclear":0.2}
    printProbRaceGivenKilled('Unarmed')


def printPartDProb():
    printBanner('d')
    p_age_given_race = probFirstGivenSecond(
            countRaceAndAge()
    )
    for race in raceQueries():
        print('%-60s %s' %(f'p(age<20 | person_was_killed_by_police=true, race={race.lower()})',
            p_age_given_race[race]['<20']))

    def printNorms(prob_dict):
        for race in raceQueries():
            print('%-60s %s' %(f'p(race={race.lower()} | person_was_killed_by_police=true, age<20)', prob_dict[race]))


    def printProbRaceGivenAge(status, suppress=False):
        p_race_killed = printProbRaceGivenKilled(status, suppress=True)
        probs = dict()
        for race in raceQueries():
            probs[race]= p_age_given_race[race]['<20'] * p_race_killed[race]
        norm_probs =  { race : probs[race] / sum(probs.values()) for race in probs}
        if not suppress:
            printNorms(norm_probs)
        return norm_probs
    
    print('\nCase 1')
    printProbRaceGivenAge('Allegedly Armed')

    print('\nCase 2')
    printProbRaceGivenAge('Unclear')
    return
    


if __name__=='__main__':
    print('Start...\n')
    INCIDENT_LIST = open_PKDv2.readPKD('assets/police_killings.csv')
    print('Incident Total: ', len(INCIDENT_LIST))

    ### Part (a) ###
    # U.S. Census race data: https://www.census.gov/library/visualizations/interactive/race-and-ethnicity-in-the-united-state-2010-and-2020-census.html
    printBanner("a")
    partA = [
    'p(white)       0.616     Census Group: White alone',
    'p(black)       0.124     Census Group: Black or African American alone',
    'p(hispanic)    0.187     Census Group: Hispanic or Latino',
    'p(asian)       0.06      Census Group: Asian alone',
    '\nTOTAL: p(white) + p(black) + p(hispanic) + p(asian) = 0.987'
    ]
    print('\n'.join(partA))
    printPartBProb()
    printPartCProb()
    printPartDProb()
    print('\nFin.')