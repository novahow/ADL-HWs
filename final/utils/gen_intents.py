import glob
import json
import sys
import argparse
from os.path import join
parser = argparse.ArgumentParser()
parser.add_argument('--negate', action='store_true')
parser.add_argument('--path', type=str)
args = parser.parse_args()
intents = {
    "Song": [],
    "Movie": [],
    "Attraction": [],
    "Transportation": [],
    "Restaurant": [],
    "Hotel": [],
} if not args.negate else {
    'None': []
}

active_intent = {
    "Hotel": ["ReserveHotel", 'SearchHouse', 'BookHouse', 'SearchHotel'],
    "Restaurant": ["FindRestaurants", 'ReserveRestaurant'],
    'Transportation': ['ReserveRoundtripFlights', 'GetRide', 
                        'ReserveCar', 'FindTrains', 'BuyBusTicket', 
                        'ReserveOnewayFlight', 'SearchOnewayFlight', 
                        'GetTrainTickets', 'SearchRoundtripFlights', 
                        'FindBus', 'GetCarsAvailable'],
    'Song': ['PlaySong', 'LookupMusic', 'LookupSong'],
    'Movie': ['GetTimesForMovie', 'RentMovie', 'BuyMovieTickets', 'FindMovies', 'PlayMovie'],
    'Attraction': ['FindAttractions'],
    } if not args.negate else {
       'None': ['RequestPayment', 'AddEvent', 'GetEventDates', 'FindProvider', 'GetAlarms', 
            'CheckBalance', 'FindApartment', 'BuyEventTickets', 'ShareLocation', 'BookAppointment',
            'GetAvailableTime', 'FindHomeByArea', 'MakePayment', 'FindEvents', 'GetWeather', 'TransferMoney',
            'AddAlarm', 'GetEvents', 'NONE']
        }
    

intent_mapper = {}
for k, v in active_intent.items():
    for i in v:
        intent_mapper[i] = k

for t in ["train", "dev", "test"]:
    dialogue_paths = glob.glob(join(args.path, f"{t}/dialogues_*"))
    for p in dialogue_paths:
        with open(p, "r") as f:
            dialogues = json.load(f)

        for d in dialogues:
            intent = None
            turns = []
            sample = {"intent_pos": 0, "dialogue": []}
            for t in range(len(d["turns"])):
                if d["turns"][t]["speaker"] == "USER":
                    for f in d["turns"][t]["frames"]:
                        if (
                            f["state"]["active_intent"] in intent_mapper.keys()
                            and intent is None
                        ):
                            intent = intent_mapper[f["state"]["active_intent"]]
                            sample["intent_pos"] = len(turns)
                        
                    turns.append(d["turns"][t]["utterance"])
                # else:
                #     turns.append(d["turns"][t]["delex"])
            if intent is not None:
                sample["dialogue"] = turns
                intents[intent].append(sample)


for (k, v) in intents.items():
    with open(f"sgd_intent_dialog/{k}_delex.json", "w") as f:
        json.dump(v, f, ensure_ascii=False, indent=4)