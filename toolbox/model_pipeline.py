import pandas.api.types as ptypes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from toolbox.data import read_data, process_roommates


def preprocess_data(config):
    # Preprocess data
    data = read_data('../data/properties.json')

    data['furnish'] = LabelEncoder().fit_transform(data['furnish'])
    data['propertyType'] = LabelEncoder().fit_transform(data['propertyType'])
    data['gender'] = LabelEncoder().fit_transform(data['gender'])
    data['matchStatus'] = LabelEncoder().fit_transform(data['matchStatus'])
    data['pets'] = LabelEncoder().fit_transform(data['pets'])
    data['roommates'] = process_roommates(data)
    data['shower'] = LabelEncoder().fit_transform(data['shower'])
    data['smokingInside'] = LabelEncoder().fit_transform(data['smokingInside'])
    data['toilet'] = LabelEncoder().fit_transform(data['toilet'])

    assert ptypes.is_numeric_dtype(data['rent'])
    assert ptypes.is_numeric_dtype(data['additionalCosts'])
    # TODO preprocessing
    return data[config['features'] + [config['y_feature']]]


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
        raise ValueError('Unknown model: {}'.format(config['model']))
    return clf


def train_model(get_model, data, config):
    X = data[config['features']]
    y = data[config['y_feature']]
    model = None
    if config['model_selection'] == 'kfold':
        kf = KFold(**config['model_selection_params'], random_state=config['random_state'])
        for train_index, test_index in kf.split(X):
            model = get_model()
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            print(model.score(X_test, y_test))
    elif config['model_selection'] == 'train_test_split':
        X_train, X_test, y_train, y_test = train_test_split(**config['model_selection_params'], random_state=config['random_state'])
        model = get_model()
        model.fit(X_train, y_train)
        print(model.score(X_test, y_test))
    else:
        raise ValueError('Unknown model selection method: {}'.format(config['model_selection']))
    return model
