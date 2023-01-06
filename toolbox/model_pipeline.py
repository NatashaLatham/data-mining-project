import pandas.api.types as ptypes
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from toolbox.data import read_data, process_roommates, process_energy_label, process_furnish, process_property_type, \
    process_gender, process_internet, process_kitchen, process_living, process_pets, process_smoking_inside, \
    process_shower, process_toilet, process_match_capacity


def preprocess_data(config):
    # Preprocess data
    data = read_data()

    if config['preprocessing'] == 'label_encoder':
        # we need the original city names for the error plots
        data['cityName'] = data['city'].copy()
        string_columns = ['city', 'furnish', 'propertyType', 'energyLabel', 'gender', 'internet', 'kitchen', 'living',
                          'matchCapacity', 'matchGender', 'matchStatus', 'pets', 'roommates', 'shower', 'smokingInside',
                          'toilet']
        data[string_columns] = data[string_columns].apply(LabelEncoder().fit_transform)
    elif config['preprocessing'] == 'custom':
        data['furnish'] = process_furnish(data)
        data['propertyType'] = process_property_type(data)
        data['energyLabel'] = process_energy_label(data)
        data['gender'] = process_gender(data)
        data['internet'] = process_internet(data)
        data['kitchen'] = process_kitchen(data)
        data['living'] = process_living(data)
        data['matchCapacity'] = process_match_capacity(data)
        data['matchStatus'] = LabelEncoder().fit_transform(data['matchStatus'])
        # we need the original city names for the error plots
        data['cityName'] = data['city'].copy()
        data['city'] = LabelEncoder().fit_transform(data['city'])
        data['pets'] = process_pets(data)
        data['roommates'] = process_roommates(data)
        data['shower'] = process_shower(data)
        data['smokingInside'] = process_smoking_inside(data)
        data['toilet'] = process_toilet(data)
    else:
        raise ValueError(f'Unknown preprocessing method: {config["preprocessing"]}')

    if config['filter_city']:
        data = data[data['cityName'] == config['city']]
        # print(f'Filtered data to city: {config["city"]}, new shape: {data.shape}')

    assert ptypes.is_numeric_dtype(data['rent'])
    assert ptypes.is_numeric_dtype(data['additionalCosts'])
    return data


def create_model(config):
    # Create model
    if config['model'] == 'decision_tree':
        clf = DecisionTreeRegressor(**config['model_params'], random_state=config['random_state'])
    elif config['model'] == 'random_forest':
        clf = RandomForestRegressor(**config['model_params'], random_state=config['random_state'])
    elif config['model'] == 'boosted_tree':
        clf = GradientBoostingRegressor(**config['model_params'], random_state=config['random_state'])
    elif config['model'] == 'mlp':
        clf = MLPRegressor(**config['model_params'], random_state=config['random_state'])
    else:
        raise ValueError(f'Unknown model: {config["model"]}')
    return clf


def train_model(get_model, data, config):
    X = data[config['features']]
    y = data[config['y_feature']]
    amsterdam_idx = data['cityName'] == 'Amsterdam'
    model = None
    if config['model_selection'] == 'kfold':
        kf = KFold(**config['model_selection_params'], random_state=config['random_state'])
        score = 0.
        average_error = 0.
        average_error_amsterdam = 0.
        for train_index, test_index in kf.split(X):
            model = get_model()
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            score += model.score(X_test, y_test)

            # calculate average error
            y_pred = model.predict(X_test)
            error = abs(y_pred - y_test)
            average_error += error.mean()

            y_pred_amsterdam = model.predict(X_test[amsterdam_idx])
            y_test_amsterdam = y_test[amsterdam_idx]
            error_amsterdam = abs(y_pred_amsterdam - y_test_amsterdam)
            average_error_amsterdam += error_amsterdam.mean()

        score /= kf.get_n_splits()
        average_error /= kf.get_n_splits()
        average_error_amsterdam /= kf.get_n_splits()
    elif config['model_selection'] == 'train_test_split':
        X_train, X_test, y_train, y_test = train_test_split(X, y, **config['model_selection_params'],
                                                            random_state=config['random_state'])
        model = get_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_error = y_pred - y_test

        y_pred_amsterdam = model.predict(X_test[amsterdam_idx])
        y_test_amsterdam = y_test[amsterdam_idx]
        y_error_amsterdam = y_pred_amsterdam - y_test_amsterdam

        # plt.hist(y_error, range=[-500, 500], bins=100)
        # plt.ylim([0, 200 if config['filter_city'] else 1000])
        sns.displot(y_error, bins=200, kde=True)
        plt.xlim([-500, 500] if config['filter_city'] else [-1000, 1000])
        plt.ylim([0, 200 if config['filter_city'] else 1000])
        plt.title(f'Error distribution {config["model"]}, {"Amsterdam" if config["filter_city"] else "Netherlands"}')
        plt.xlabel('Error €')
        plt.ylabel('Count')
        plt.tight_layout()
        if config['save_error_plot']:
            plt.savefig(f'plots/error_distribution_{config["model"]}_'
                        f'{"amsterdam" if config["filter_city"] else "netherlands"}.pdf')
        plt.show()

        average_error = abs(y_error).mean()
        average_error_amsterdam = abs(y_error_amsterdam).mean()

        score = model.score(X_test, y_test)
    else:
        raise ValueError('Unknown model selection method: {}'.format(config['model_selection']))
    return model, score, average_error, average_error_amsterdam
