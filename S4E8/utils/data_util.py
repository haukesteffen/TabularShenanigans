import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

numerical_features = [
    'cap-diameter',
    'stem-height',
    'stem-width'
]

categorical_features = [
    'cap-shape',
    'cap-surface',
    'cap-color',
    'does-bruise-or-bleed',
    'gill-attachment',
    'gill-spacing',
    'gill-color',
    'stem-root',
    'stem-surface',
    'stem-color',
    'veil-type',
    'veil-color',
    'has-ring',
    'ring-type',
    'spore-print-color',
    'habitat',
    'season'
]

class MushroomDataset(Dataset):
    def __init__(self, n_bins, subset='train', preprocessors=None, val_size=0.1):
        self.subset = subset
        assert self.subset in ['train', 'val', 'test']
        if self.subset in ['train', 'val']:
            self.df = pd.read_csv('train.csv')
        else:
            self.df = pd.read_csv('test.csv')

        train, val = train_test_split(
            self.df,
            test_size=val_size,
            shuffle=False,
            random_state=42
            )  

        if subset == 'train':
            # preprocess features
            self.numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('kbins', KBinsDiscretizer(encode='ordinal', strategy='quantile', n_bins=n_bins))
            ])
            self.categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(max_categories=n_bins-1, handle_unknown='use_encoded_value', unknown_value=n_bins))
            ])
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', self.numerical_pipeline, numerical_features),
                    ('cat', self.categorical_pipeline, categorical_features)
                ]
            )
            self.X = torch.tensor(self.preprocessor.fit_transform(train[numerical_features + categorical_features])).int()

            # preprocess labels
            self.label_enc = LabelEncoder()
            label_array = self.label_enc.fit_transform(train['class'])
            self.y = torch.tensor(label_array).float()

        if subset == 'val':
            assert preprocessors is not None
            self.preprocessors = preprocessors
            self.X = torch.tensor(self.preprocessors[0].transform(val[numerical_features + categorical_features])).int()
            label_array = self.preprocessors[1].transform(val['class'])
            self.y = torch.tensor(label_array).float()

        if subset == 'test':
            assert preprocessors is not None
            self.preprocessors = preprocessors
            self.X = torch.tensor(self.preprocessors[0].transform(self.df[numerical_features + categorical_features])).int()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.subset in ['train', 'val']:
            output = (self.X[idx], self.y[idx])
        if self.subset == 'test':
            output = self.X[idx]
        return output