import geopandas as gpd
import pandas as pd
import json
import math

def read_coastline(path='data/2019_ggd_regios_kustlijn.gpkg'):
    """Load data from GeoPackage and return a GeoDataFrame"""
    return gpd.read_file(path)


def read_geojson(path='data/GEBIED_BUURTCOMBINATIES_EXWATER.geojson'):
    """Load data from GeoJSON and return a GeoDataFrame"""
    return gpd.read_file(path)


def to_date_series(series):
    """Convert a series to a date series"""
    # date format: {'$date': '2019-07-24T22:25.099+0000'}
    series = series.apply(lambda x: x['$date'])
    return pd.to_datetime(series)


def read_data(path='data/properties.json'):
    """Load data from JSON and return a DataFrame"""
    data = pd.read_json(path, lines=True)
    date_columns = ['crawledAt', 'firstSeenAt', 'lastSeenAt', 'detailsCrawledAt']
    for column in date_columns:
        data[column] = to_date_series(data[column])
    data['_id'] = data['_id'].apply(lambda x: x['$oid'])
    return data

def process_roommates(input_data):
    cleaned = input_data["roommates"].copy()
    for i, roommate in enumerate(input_data["roommates"]):
        if type(roommate) == str or roommate is None:
            if roommate is None or roommate == 'None':
                cleaned[i] = 0
            elif roommate.isnumeric():
                cleaned[i] = int(roommate)
            elif roommate == "More than 8":
                cleaned[i] = 9
            else:
                # happens when roommate is 'Unknown'
                cleaned[i] = -1
        else:
            # happens when roommate is nan
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_energy_label(input_data):
    cleaned = input_data["energyLabel"].copy()
    labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    for i, label in enumerate(input_data["energyLabel"]):
        if type(label) == str and label != 'Unknown':
            cleaned[i] = labels[label]
        else:
            # happens when roommate is nan or Unknown
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_external_id(input_data):
    # has value room, studio, apartment, anti? (336 records), student%20residence? (5 records)
    return [x.split('-')[0] for x in input_data['externalId']]

def process_furnished(input_data):
    # TODO: check what the empty means, perhaps make into own category
    return [x for x in input_data['furnished'] if x]


def process_gender(input_data):
    cleaned = input_data["gender"].copy()
    genders = {'Mixed': 0, 'Female': 1, 'Male': 2}
    for i, gender in enumerate(input_data["gender"]):
        if type(gender) == str and gender != 'Unknown':
            cleaned[i] = genders[gender]
        else:
            # happens when gender is nan, None or Unknown
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_internet(input_data):
    # can be Yes, Unknown, No or nan
    cleaned = input_data["internet"].copy()
    internet_option = {'No': 0, 'Yes': 1}
    for i, option in enumerate(input_data["internet"]):
        if type(option) == str and option != 'Unknown':
            cleaned[i] = internet_option[option]
        else:
            # happens when internet is nan or Unknown
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_is_room_active(input_data):
    # can be true, false or nan
    # TODO: what does this actually mean?
    cleaned = input_data["isRoomActive"].copy()
    internet_option = {'false': 0, 'true': 1}
    for i, option in enumerate(input_data["isRoomActive"]):
        if type(option) == str:
            cleaned[i] = internet_option[option]
        else:
            # happens when internet is nan
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_kitchen_or_living(input_data, key):
    # can be true, false or nan
    cleaned = input_data[key].copy()
    options = {'Shared': 0, 'Own': 1}
    for i, option in enumerate(input_data[key]):
        if type(option) == str and option not in ['Unknown', 'None']:
            cleaned[i] = options[option]
        else:
            # happens when option is nan
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_kitchen(input_data):
    return process_kitchen_or_living(input_data, 'kitchen')
def process_living(input_data):
    return process_kitchen_or_living(input_data, 'living')

def process_match_capacity(input_data):
    # can be 1 person .. 5 persons, nan, > 5 persons, Not important
    key = 'matchCapacity'
    cleaned = input_data[key].copy()

    for i, option in enumerate(input_data[key]):
        if type(option) == str:
            split_data = option.split(' ')[0]

            if split_data.isnumeric():
                cleaned[i] = int(split_data)
            elif split_data == '>':
                cleaned[i] = 6
            else:
                #happens when Not important
                cleaned[i] = 0
        else:
            # happens when option is nan
            cleaned[i] = -1
    return cleaned.astype("int64")

def process_pets(input_data):
    # can be No, Yes, nan, By mutual agreement
    key = 'pets'
    cleaned = input_data[key].copy()
    options = {'No': 0, 'Yes': 1, 'By mutual agreement':2}
    for i, option in enumerate(input_data[key]):
        if type(option) == str:
            cleaned[i] = options[option]
        else:
            # happens when internet is nan or Unknown
            cleaned[i] = -1
    return cleaned.astype("int64")

if __name__ == '__main__':
    data = read_data('../data/properties.json')
    # print(type(data["roommates"][96]))
    data["pets_cleaned"] = process_pets(data)
    # [i for i, x in enumerate(data['gender']) if x is None]
    # ext_Id = [x.split('-')[0] for x in data['externalId']]
    print(data.shape)
