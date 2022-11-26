import geopandas as gpd
import pandas as pd


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


if __name__ == '__main__':
    read_data('../data/properties.json')
