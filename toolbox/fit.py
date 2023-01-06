import os
from itertools import product

from tqdm import tqdm
import pandas as pd

from toolbox.model_pipeline import preprocess_data, create_model, train_model

CONFIG = {
    # preprocessing
    # 'preprocessing': 'label_encoder',
    'preprocessing': 'custom',

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
        'city',
        # 'deposit',
        'energyLabel',
        'furnish',
        'gender',
        'internet',
        'kitchen',
        'living',
        'matchCapacity',
        'matchStatus',
        'pets',
        'propertyType',
        # 'rent',
        'roommates',
        'shower',
        'smokingInside',
        'toilet',
    ],

    'filter_city': False,
    # 'filter_city': True,
    # 'city': 'Amsterdam',

    'save_error_plot': True,
    # 'save_error_plot': False,

    'random_state': 42,

    # 'model_selection': 'kfold',
    # 'model_selection_params': {
    #     'n_splits': 5,
    #     'shuffle': True,
    # },

    'model_selection': 'train_test_split',
    'model_selection_params': {
        'test_size': 0.2,
        'shuffle': True,
    },
}


def pipeline(config):
    data = preprocess_data(config)
    model, score, average_error, average_error_amsterdam = train_model(lambda: create_model(config), data, config)
    return score, average_error, average_error_amsterdam


if __name__ == '__main__':
    pipeline(CONFIG)
    results = []
    # for use_amsterdam, (model, params) in tqdm(list(product([True, False], [
    #     ('decision_tree', {}),
    #     ('random_forest', {}),
    #     ('boosted_tree', {}),
    #     ('mlp', {}),
    # ]))):    
    for use_amsterdam, (model, params) in list(product([True, False], [
        ('decision_tree', {}),
        ('random_forest', {}),
        ('boosted_tree', {}),
        ('mlp', {}),
    ])):
    
        config = CONFIG.copy()
        config['filter_city'] = use_amsterdam
        config['city'] = 'Amsterdam'
        config['model'] = model
        config['model_params'] = params
        score, average_error, average_error_amsterdam = pipeline(config)
        results.append({
            'use_amsterdam': use_amsterdam,
            'model': model,
            'score': score,
            'average_error': average_error,
            'average_error_amsterdam': average_error_amsterdam,
        })
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_parquet('results/model_scores.parquet')
    print(results_df)
