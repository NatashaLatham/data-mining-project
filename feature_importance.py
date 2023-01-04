import os

import pandas as pd

from toolbox.fit import CONFIG
from toolbox.model_pipeline import preprocess_data, train_model, create_model

if __name__ == '__main__':
    config = CONFIG.copy()
    config['model'] = 'random_forest'
    config['model_params'] = {}

    data = preprocess_data(config)
    model, score = train_model(lambda: create_model(config), data, config)
    importances = []
    for feature in config['features']:
        importances.append({'feature': feature,
                            'score': model.feature_importances_[config['features'].index(feature)]})
    importances_df = pd.DataFrame(importances)
    print(importances_df)
    os.makedirs('results', exist_ok=True)
    importances_df.to_parquet('results/feature_importances.parquet')
