import os

import numpy as np
import pandas as pd

from toolbox.fit import CONFIG
from toolbox.model_pipeline import preprocess_data, train_model, create_model

if __name__ == '__main__':
    config = CONFIG.copy()
    config['model'] = 'random_forest'
    config['model_params'] = {}

    data = preprocess_data(config)
    model, score = train_model(lambda: create_model(config), data, config)
    feature_importances = []
    for feature in config['features']:
        importances = model.feature_importances_[config['features'].index(feature)]
        std = np.std([tree.feature_importances_[config['features'].index(feature)] for tree in model.estimators_])
        feature_importances.append({'feature': feature,
                                    'importance': importances,
                                    'std': std})
    importances_df = pd.DataFrame(feature_importances)
    print(importances_df)
    os.makedirs('results', exist_ok=True)
    importances_df.to_parquet('results/feature_importances.parquet')
