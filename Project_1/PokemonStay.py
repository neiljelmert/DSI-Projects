
from pprint import pprint

#                    PROJECT 1
########################################################################################################
########################################################################################################


# 1, 2, 3
player1 = {"player_id": 1, "player_name": "Ash", "time_played": 122.5,
          "player_pokemon": {}, "gyms_visited": ["netflix.com", "linkedin.com"]}

player2 = {"player_id": 2, "player_name": "Beans", "time_played": 732.6,
          "player_pokemon": {}, "gyms_visited": ["stackoverflow.com", "github.com"]}


gym_locs = ["reddit.com", "amazon.com", "twitter.com", "linkedin.com",
            "ebay.com", "netflix.com", "facebook.com", "stackoverflow.com",
            "github.com", "quora.com"]

dict = {"pokemon_id": int, "name": str, "type": str, "hp": int, "attack": int, "defense": int,
           "special_attack": int, "special_defense": int, "speed": int}

pokedex = {
            1: {
                    "name": "charmander",
                    "type": "fire",
                    "hp": 20,
                    "attack": 22,
                    "defense": 15,
                    "special_attack": 10,
                    "special_defense": 11,
                    "speed": 28
            },
            2: {
                    "name": "squirtle",
                    "type": "water",
                    "hp": 18,
                    "attack": 12,
                    "defense": 30,
                    "special_attack": 25,
                    "special_defense": 40,
                    "speed": 5
            },
            3: {    "name": "bulbasaur",
                    "type": "poison",
                    "hp": 18,
                    "attack": 20,
                    "defense": 25,
                    "special_attack": 15,
                    "special_defense": 35,
                    "speed": 10
            }
         }

########################################################################################################

# 4 and 5
players = {}
player1.pop("player_id", None)
player2.pop("player_id", None)
players[1] = player1
players[2] = player2
players[1]["player_pokemon"] = {2: pokedex[2]}
players[2]["player_pokemon"] = {1: pokedex[1], 3: pokedex[3]}

print "DATA FOR 4 AND 5"
pprint(players)

#OUTPUT
'''
{1: {'gyms_visited': ['netflix.com', 'linkedin.com'],
     'player_name': 'Ash',
     'player_pokemon': {2: {'attack': 12,
                            'defense': 30,
                            'hp': 18,
                            'name': 'squirtle',
                            'special_attack': 25,
                            'special_defense': 40,
                            'speed': 5,
                            'type': 'water'}},
     'time_played': 122.5},
 2: {'gyms_visited': ['stackoverflow.com', 'github.com'],
     'player_name': 'Beans',
     'player_pokemon': {1: {'attack': 22,
                            'defense': 15,
                            'hp': 20,
                            'name': 'charmander',
                            'special_attack': 10,
                            'special_defense': 11,
                            'speed': 28,
                            'type': 'fire'},
                        3: {'attack': 20,
                            'defense': 25,
                            'hp': 18,
                            'name': 'bulbasaur',
                            'special_attack': 15,
                            'special_defense': 35,
                            'speed': 10,
                            'type': 'poison'}},
     'time_played': 732.6}}
'''

########################################################################################################

# 6
for gym in gym_locs:
    for player in players:
        if gym in players[player]["gyms_visited"]:
            print "DATA FOR 6", str(players[player]["player_name"]) + " has visited " + str(gym)


#OUTPUT
'''
Ash has visited linkedin.com
Ash has visited netflix.com
Beans has visited stackoverflow.com
Beans has visited github.com
'''

########################################################################################################

# 7
def power(players, pokedex, player_id):

    for i in pokedex.keys():
        if i in players[player_id]["player_pokemon"].keys():
             pwr = sum([x for x in pokedex[i].values() if type(x) == int])
             print "DATA FOR 7", str(players[player_id]["player_name"]) + "'s power is " + str(pwr)

power(players, pokedex, 2)

#OUTPUT
'''
Beans's power is 106
Beans's power is 123
'''

########################################################################################################

# 8.1
pokedex_file = "/Users/ga/Desktop/DSI-SF-3/datasets/pokemon/pokedex_basic.csv"
with open(pokedex_file, "r") as f:
    newdata = []
    raw_pd = f.read()
    raw_pd = raw_pd.split("\n")
    for strings in raw_pd:
        data = []
        strings = strings.split(",")
        for x in strings:
            x = x[1:-1]
            if x.isdigit():
                 x = float(x)
            data.append(x)
        newdata.append(data)
    print "DATA FOR 8.1", newdata


#OUTPUT (first three lists)
'''
[['PokedexNumber', 'Name', 'Type', 'Total', 'HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed'],
[1.0, 'Bulbasaur', 'GrassPoison', 318.0, 45.0, 49.0, 49.0, 65.0, 65.0, 45.0],
[2.0, 'Ivysaur', 'GrassPoison', 405.0, 60.0, 62.0, 63.0, 80.0, 80.0, 60.0],
[3.0, 'Venusaur', 'GrassPoison', 525.0, 80.0, 82.0, 83.0, 100.0, 100.0, 80.0], ...
'''

########################################################################################################

# 8.2
pokedex_file = "/Users/ga/Desktop/DSI-SF-3/datasets/pokemon/pokedex_basic.csv"
with open(pokedex_file, "r") as f:
    raw_pd_82 = f.read()
    raw_pd_82 = raw_pd_82.split("\n")

    newdata_82 = [[float(x[1:-1]) if x[1:-1].isdigit() else str(x[1:-1]) for x in strings.split(",")] for strings in raw_pd_82]
    #newdata_82 = map(lambda k: list(newdata_82[10 * k: 10 * (k + 1)]), range(0, len(newdata_82) / 10))
    # my other hacky way ^ i liked it =]
    print "DATA FOR 8.2", newdata_82

# NOTE: THERE ARE NO EMPTY VALUES -- check the following code

#type_list = [float, str, str, float, float, float, float, float, float, float]
#pokedex_file = "/Users/ga/Desktop/DSI-SF-3/datasets/pokemon/pokedex_basic.csv"
#with open(pokedex_file, "rb") as f:
    #reader = csv.reader(f)
    #next(reader)
    #for row in reader:
        #typed_row = [func(val) for func, val in zip(type_list, row)]
        #print typed_row  #, ' ' or '' in typed_row ----- NO EMPTIES

#OUTPUT (first three lists)
'''
[['PokedexNumber', 'Name', 'Type', 'Total', 'HP', 'Attack', 'Defense', 'SpecialAttack', 'SpecialDefense', 'Speed'],
[1.0, 'Bulbasaur', 'GrassPoison', 318.0, 45.0, 49.0, 49.0, 65.0, 65.0, 45.0],
[2.0, 'Ivysaur', 'GrassPoison', 405.0, 60.0, 62.0, 63.0, 80.0, 80.0, 60.0],
[3.0, 'Venusaur', 'GrassPoison', 525.0, 80.0, 82.0, 83.0, 100.0, 100.0, 80.0], ...
'''

########################################################################################################

# 9
def pokegen(newdata, pokemon_id):
    dict = {}

    id = float(pokemon_id)
    for list in newdata:
        if list[0] == id:
            #zip(newdata[0], list)
            dict = {k: v for (k,v) in zip(newdata[0], list)}
            print "DATA FOR 9"
            pprint(dict)

pokegen(newdata, 720)

#OUTPUT
'''
{'Attack': 110.0,
 'Defense': 60.0,
 'HP': 80.0,
 'Name': 'HoopaHoopa Confined',
 'PokedexNumber': 720.0,
 'SpecialAttack': 150.0,
 'SpecialDefense': 130.0,
 'Speed': 70.0,
 'Total': 600.0,
 'Type': 'PsychicGhost'}

{'Attack': 160.0,
 'Defense': 60.0,
 'HP': 80.0,
 'Name': 'HoopaHoopa Unbound',
 'PokedexNumber': 720.0,
 'SpecialAttack': 170.0,
 'SpecialDefense': 130.0,
 'Speed': 80.0,
 'Total': 680.0,
 'Type': 'PsychicDark'}
'''

########################################################################################################

# 10
def filter_pokedex(newdata, myfilter):

    alldict = []

    myfilter_options = myfilter.keys()

    for list in newdata[1:-1]:
        dict = {k: v for (k, v) in zip(newdata[0], list)}
        alldict.append(dict)

    for dict in alldict:
        bool_list = []
        for option in myfilter_options:
            if type(myfilter[option]) == int:
                type(myfilter[option]) == float
                bool_list.append(myfilter[option] <= dict[option])
            else:
                bool_list.append(myfilter[option] == dict[option])

        if all(bool_list):
            print "DATA FOR 10", dict


example = {
    'Attack':  100,
    'Type': 'Rock',
    'Defense': 115
}

filter_pokedex(newdata, example)

#OUTPUT
'''
{'Name': 'Sudowoodo', 'SpecialAttack': 30.0, 'HP': 70.0, 'Speed': 30.0, 'Attack': 100.0, 'Defense': 115.0, 'PokedexNumber': 185.0, 'SpecialDefense': 65.0, 'Total': 410.0, 'Type': 'Rock'}
{'Name': 'Regirock', 'SpecialAttack': 50.0, 'HP': 80.0, 'Speed': 50.0, 'Attack': 100.0, 'Defense': 200.0, 'PokedexNumber': 377.0, 'SpecialDefense': 100.0, 'Total': 580.0, 'Type': 'Rock'}
{'Name': 'Gigalith', 'SpecialAttack': 60.0, 'HP': 85.0, 'Speed': 25.0, 'Attack': 135.0, 'Defense': 130.0, 'PokedexNumber': 526.0, 'SpecialDefense': 80.0, 'Total': 515.0, 'Type': 'Rock'}
'''

########################################################################################################
########################################################################################################

'''
list1 = [1.5, 3.5, 5.5, 7.5]
list2 = [0, 4, 8, 12]


def multiply_list(list1, list2, n = 5000):
    count = 0
    print list1, list2
    multiplied = 0
    while multiplied <= n:
        count += 1
        print "Iteration: ", count
        for value1, value2 in zip(list1, list2):
            print value1, value2
            multiplied = value1 * value2
            print multiplied
        if multiplied > n:
            break
        else:
            list1 = [2*x for x in list1]
            list2 = [2*y for y in list2]
            print list1, list2

multiply_list(list1, list2)'''