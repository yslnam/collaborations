# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from scipy import stats
import category_encoders as ce

def clean_dataframe(hdf):
    ord_feat_cat = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
            'HeatingQC', 'KitchenQual', 'FireplaceQu', 
            'GarageQual', 'GarageCond', 'PoolQC']

    # Transform ordinary features in numeric
    hdf[ord_feat_cat] = hdf[ord_feat_cat].replace({'Ex': 5, 'Gd': 4, 
                                                'TA': 3, 'Fa': 2, 
                                                'Po': 1, np.nan: 0})

    hdf[['BsmtExposure']] = hdf[['BsmtExposure']].replace({'Gd': 4, 'Av': 3, 
                                                        'Mn': 2, 'No': 1, 
                                                        np.nan: 0})

    hdf[['BsmtFinType1', 'BsmtFinType2']] = hdf[['BsmtFinType1', 
                                                'BsmtFinType2']].replace({'GLQ': 6, 'ALQ': 5, 
                                                                        'BLQ': 4, 'Rec': 3, 
                                                                        'LwQ': 2, 'Unf': 1, 
                                                                        np.nan: 0})

    hdf[['Fence']] = hdf[['Fence']].replace({'MnPrv': 1, 'GdWo': 1, 
                                            'GdPrv': 1, 'MnWw': 1, 
                                            np.nan: 0})

    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
        hdf[col].fillna(0, inplace = True)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        hdf[col].fillna('NoGarage', inplace = True)
    for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
        hdf[col].fillna('NoBsmt', inplace = True)

    # Input LotFrontage with the median of neighborhood
    hdf['LotFrontage'] = hdf.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Inpute all the rest na's with median or mode.
    for col in set(hdf.columns) - {'SalePrice'}:
        if hdf[col].dtype == 'object':
            hdf.fillna({col:stats.mode(hdf[col]).mode[0]}, inplace = True)
        else:
            hdf.fillna({col:np.median(hdf.loc[~hdf[col].isnull(), col])}, inplace = True)
    return hdf

def Feature_enginiering(hdf):
    ord_feat_num = ['OverallQual', 'OverallCond', 'BsmtFullBath', 
            'BsmtHalfBath', 'FullBath', 'HalfBath',
            'TotRmsAbvGrd', 'Fireplaces', 'BedroomAbvGr', 
            'KitchenAbvGr', 'GarageCars']
    ord_feat_cat = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
            'HeatingQC', 'KitchenQual', 'FireplaceQu', 
            'GarageQual', 'GarageCond', 'PoolQC']
    nom_feat = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 
        'LandContour', 'Utilities', 'Neighborhood', 
        'Condition1', 'Condition2', 'BldgType', 'RoofStyle', 
        'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
        'Foundation', 'Heating', 'CentralAir', 'Electrical', 
        'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition', 
        'GarageFinish', 'PavedDrive', 'Fence', 'Functional', 
        'HouseStyle','LotConfig', 'Street', 'LandSlope']
    cont_feat = ['LotFrontage', 'LotArea', 'YearBuilt', 
        'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 
        'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
        'GarageYrBlt', 'WoodDeckSF', 'OpenPorchSF', 
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
        'MiscVal', 'MoSold', 'YrSold','BsmtUnfSF', 'GarageArea', 
        'LowQualFinSF', 'GrLivArea']

    hdf['Remodeled'] = pd.Series([1 if a > 0 else 0 for a in (hdf['YearRemodAdd'] - hdf['YearBuilt'])])
    hdf['TotalPorchAreasSF'] = hdf['OpenPorchSF'] + hdf['EnclosedPorch'] + \
                                hdf['3SsnPorch'] + hdf['ScreenPorch'] + hdf['WoodDeckSF']
    hdf ['TotalBath'] = hdf['FullBath'] + hdf['BsmtFullBath'] + .5*(hdf['HalfBath'] + hdf['BsmtHalfBath'])
    hdf['OtherRoomsAbvGrd'] = hdf['TotRmsAbvGrd'] - hdf['KitchenAbvGr'] - hdf['FullBath']

    hdf['HasPool'] = hdf['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    hdf['Has2ndFloor'] = hdf['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    hdf['HasGarage'] = hdf['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    hdf['HasBsmt'] = hdf['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    hdf['HasFireplace'] = hdf['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


    hdf.drop(columns = ['Utilities', 'Street', 'PoolQC'], inplace = True)
    hdf.drop(columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], inplace = True)
    hdf.drop(columns = ['FullBath', 'BsmtFullBath', 'HalfBath','BsmtHalfBath'], inplace = True)
    hdf.drop(columns = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF'], inplace = True)
    hdf.drop(columns = 'MiscFeature', inplace = True)
    hdf.drop(columns = 'TotRmsAbvGrd', inplace = True)

    ord_feat_num = set(ord_feat_num).union(set(['TotalBath', 'OtherRoomsAbvGrd', 'Remodeled'])) - \
    set(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',' KitchenAbvGr', 'GarageCars'])

    ord_feat_cat = set(ord_feat_cat) - set(['PoolQC'])

    nom_feat = set(nom_feat).union(['HasPool', 'Has2ndFloor', 'HasGarage', 'HasGarage', 'HasBsmt', 'HasFireplace']) - set(['Utilities','MiscFeature','Street'])

    cont_feat = set(cont_feat).union(set(['TotalPorchAreasSF'])) - set(['Utilities', 'Street', 'PoolQC'] +\
                                                                                                              ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'] +\
                                                                                                              ['FullBath', 'BsmtFullBath', 'HalfBath','BsmtHalfBath'] +\
                                                                                                              ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF'] +\
                                                                                                              ['MiscFeature', 'TotRmsAbvGrd'])
    return hdf, ord_feat_num, ord_feat_cat, nom_feat, cont_feat

def Train_test_random_forest(hdf, ord_feat_num, ord_feat_cat, nom_feat, cont_feat):
    ord_feat = ord_feat_cat.union(ord_feat_num)
    X_train = hdf.loc[~hdf['SalePrice'].isnull(), :]

    y_train = np.log1p(X_train.loc[~X_train['SalePrice'].isnull(), 'SalePrice'])

    X_train.drop(columns = ['SalePrice'], inplace = True)

    X_test = hdf.loc[hdf['SalePrice'].isnull(), :].drop(columns = ['SalePrice'])

    ord_enc = ce.OrdinalEncoder(cols=ord_feat).fit(X_train,y_train)
    X_train = ord_enc.transform(X_train)

    X_test = ord_enc.transform(X_test)

    perm = np.random.permutation(len(X_train))
    X_train = X_train.iloc[perm].reset_index(drop=True)
    y_train = y_train.iloc[perm].reset_index(drop=True)

    nom_enc = ce.CatBoostEncoder(cols=nom_feat).fit(X_train,y_train)
    X_train = nom_enc.transform(X_train)

    X_test = nom_enc.transform(X_test)

    X_train.drop(columns = ['Id'], inplace = True)
    return X_train, y_train, X_test

def Train_test_normalized(hdf, ord_feat_num, ord_feat_cat, nom_feat, cont_feat):
    # Dummify and Transforming prine to log price
    X = pd.get_dummies(hdf, columns = nom_feat, drop_first=True)


    for col in set(X.columns) - {'SalePrice', 'Id'}:
        if np.std(X[col]) != 0:
            X.loc[:, col] = (X[col] - np.mean(X[col]))/np.std(X[col], ddof=1)

    X_train = X.loc[~X['SalePrice'].isnull(), :]

    X_train.drop((X_train.loc[X_train['GrLivArea']>4.3, :]).index, inplace = True)

    X_train.drop((X_train.loc[X_train['LotArea']>10, :]).index, inplace = True)

    X_train.drop((X_train.loc[X_train['LotFrontage']>6, :]).index, inplace = True)

    X_train.reset_index(drop=True, inplace=True)

    y_train = np.log1p(X_train.loc[~X_train['SalePrice'].isnull(), 'SalePrice']) ## log price here!!

    X_train.drop(columns = ['Id', 'SalePrice'], inplace = True)
    X_test = X.loc[X['SalePrice'].isnull(), :].drop(columns = ['SalePrice'])
    return X_train, y_train, X_test


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    housing_prices_df_raw = pd.read_csv(f'{input_filepath}/train.csv')
    test_housing_prices_df_raw = pd.read_csv(f'{input_filepath}/test.csv')

    train_hdf = housing_prices_df_raw.copy()
    test_hdf = test_housing_prices_df_raw.copy()

    hdf = pd.concat([train_hdf, test_hdf], axis = 0, sort = False)

    clean_hdf = clean_dataframe(hdf)

    final_df, ord_feat_num, ord_feat_cat, nom_feat, cont_feat = Feature_enginiering(clean_hdf)

    X_train_rf, y_train_rf, X_test_rf = Train_test_random_forest(final_df, ord_feat_num, ord_feat_cat, nom_feat, cont_feat)

    X_train_rf.to_csv(f'{output_filepath}/X_train_rf.csv', index = False)
    y_train_rf.to_csv(f'{output_filepath}/y_train_rf.csv', index = False, header = 'SalePrice')
    X_test_rf.to_csv(f'{output_filepath}/X_test_rf.csv', index = False)

    X_train_norm, y_train_norm, X_test_norm = Train_test_normalized(final_df, ord_feat_num, ord_feat_cat, nom_feat, cont_feat)

    X_train_norm.to_csv(f'{output_filepath}/X_train_norm.csv', index = False)
    y_train_norm.to_csv(f'{output_filepath}/y_train_norm.csv', index = False, header = 'SalePrice')
    X_test_norm.to_csv(f'{output_filepath}/X_test_norm.csv', index = False)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
