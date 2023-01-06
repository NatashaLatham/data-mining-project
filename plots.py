import os
from itertools import product

from matplotlib import pyplot as plt

from toolbox.data import read_data, read_coastline, read_geojson


def plot_map(dataframe, column, title=None, cmap=None, figsize=(9, 7), colorbar_axes=None, path=None):
    if cmap is None:
        cmap = 'plasma'
    if colorbar_axes is None:
        colorbar_axes = [.81, .15, .04, .7]

    f, ax = plt.subplots(figsize=figsize)
    dataframe.plot(ax=ax, column=column, edgecolor='grey', cmap=cmap)
    if title is not None:
        ax.set_title(title)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    cax = f.add_axes(colorbar_axes)
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap,
                                            norm=plt.Normalize(vmin=gdf[column].min(),
                                                               vmax=gdf[column].max()))
    f.colorbar(scalar_mappable, cax=cax)
    if path is not None:
        plt.savefig(path)


def generate_gdf(use_amsterdam, data):
    if use_amsterdam:
        gdf = read_geojson(crs=3857)
    else:
        gdf = read_coastline(crs=3857)

    for i, row in gdf.iterrows():
        data_within_region = data[data.within(row.geometry)]

        gdf.loc[i, 'advertisements'] = len(data_within_region)
        gdf.loc[i, 'median_rent'] = data_within_region['rent'].median()
        gdf.loc[i, 'median_area'] = data_within_region['areaSqm'].median()
        gdf.loc[i, 'mean_rent'] = data_within_region['rent'].mean()
        gdf.loc[i, 'mean_area'] = data_within_region['areaSqm'].mean()
        gdf.loc[i, 'mean_rent_per_sqm'] = (
                data_within_region['rent'] / data_within_region['areaSqm']).mean()
    return gdf


if __name__ == '__main__':
    # we need to convert it to a different coordinate reference system to be able to use the within method
    data = read_data(crs=3857)
    coastline = generate_gdf(use_amsterdam=False, data=data)
    amsterdam = generate_gdf(use_amsterdam=True, data=data)
    for gdf in [coastline, amsterdam]:
        gdf.plot(color='lightgrey', edgecolor='grey')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.show()

    os.makedirs('plots', exist_ok=True)
    for use_amsterdam, (column, title) in product([True, False], [
        ('advertisements', 'Number of advertisements'),
        ('median_rent', 'Median rent'),
        ('median_area', 'Median area'),
        ('mean_rent', 'Mean rent'),
        ('mean_area', 'Mean area'),
        ('mean_rent_per_sqm', 'Mean rent per square meter'),
    ]):
        gdf = amsterdam if use_amsterdam else coastline
        plot_map(gdf, column, title=title,
                 path=f'plots/{column}_{"amsterdam" if use_amsterdam else "netherlands"}.pdf',
                 colorbar_axes=[.91, .15, .04, .7] if use_amsterdam else [.78, .15, .04, .7])
        plt.show()
