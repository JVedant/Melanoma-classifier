import os
import pandas as pd
from config.config import *

from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    input_path = TRAIN_DATASET
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = StratifiedKFold(n_splits=FOLDS)

    for folds_, (_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, "kfold"] = folds_

    #df.to_csv(os.path.join(input_path, 'train_folds.csv'), index=False)
    df.to_csv(TRAIN_FOLDS)