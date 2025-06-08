import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

numerical_features = [
    'milage',
    'model_age',
    'median_brand_price',
    'brand_popularity',
    'horse_power',
    'engine_displacement',
    'n_cylinders',
    'n_valves'
]

categorical_features = [
    'fuel_type',
    'ext_col',
    'int_col',
    'accident',
    'clean_title',
    'automatic_transmission',
    'is_v_type'
]

target = 'price'

def preprocess(df):
    df['model_age'] = 2024 - df['model_year']
    df['median_brand_price'] = df.groupby('brand')['price'].transform('median')
    df['brand_popularity'] = df.groupby('brand')['id'].transform('count') / len(df)
    df['automatic_transmission'] = df['transmission'].str.extract(r'\b(A/T|Automatic|CVT|Variable|AT)\b').notna().astype(float)
    df['horse_power'] = df['engine'].str.extract(r'(\d+\.\d+)(?=HP)').astype(float)
    df['engine_displacement'] = df['engine'].str.extract(r'(\d+\.\d+)\s?(?:L|Liter)').astype(float)
    df['n_cylinders'] = df['engine'].str.extract(r'(\d+)\s?Cylinder|V(\d+)', expand=False).iloc[:, 0].astype(float)
    df['is_v_type'] = df['engine'].str.contains(r'\bV\d+\b').astype(float)
    df['n_valves'] = df['engine'].str.extract(r'(\d+)(?=V\s)').astype(float)
    return df

class CarsDataset(Dataset):
    def __init__(self, n_bins, subset='train', preprocessors=None, val_size=0.1):
        self.subset = subset
        assert self.subset in ['train', 'val', 'test']
        if self.subset in ['train', 'val']:
            df = pd.read_csv('train.csv')
        else:
            df = pd.read_csv('test.csv')

        self.df = preprocess(df)

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
            self.scaler = StandardScaler()
            y = self.scaler.fit_transform(train[[target]])
            self.y = torch.tensor(y).float().squeeze()

        if subset == 'val':
            assert preprocessors is not None
            self.preprocessors = preprocessors
            self.X = torch.tensor(self.preprocessors[0].transform(val[numerical_features + categorical_features])).int()
            y = self.preprocessors[1].transform(val[[target]])
            self.y = torch.tensor(y).float().squeeze()

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