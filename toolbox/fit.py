from toolbox.data import read_data
from toolbox.model_pipeline import preprocess_data, create_model, train_model

CONFIG = {
    # 'model': 'decision_tree',
    # 'model_params': {},

    'model': 'random_forest',
    'model_params': {},

    # 'model': 'boosted_tree',
    # 'model_params': {},

    # 'model': 'mlp',
    # 'model_params': {},
    'y_feature': 'rent',

    'features': [
        # 'additionalCosts',
        'areaSqm',
        # 'city',
        # 'deposit',
        'energyLabel',
        'furnish',
        'gender',
        'internet',
        'kitchen',
        'living',
        'matchCapacity',
        # 'matchStatus',
        'pets',
        'propertyType',
        # 'rent',
        'roommates',
        'shower',
        'smokingInside',
        'toilet',
    ],

    'random_state': 42,

    'model_selection': 'kfold',
    'model_selection_params': {
        'n_splits': 5,
        'shuffle': True,
    },

    # 'model_selection': 'train_test_split',
    # 'model_selection_params': {
    #     'test_size': 0.2,
    #     'shuffle': True,
    # },
}


if __name__ == '__main__':
    data = preprocess_data(CONFIG)
    train_model(lambda: create_model(CONFIG), data, CONFIG)

