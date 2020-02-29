import pandas as pd
import numpy as np
from scipy import stats
import category_encoders as ce


def read_dataframe():
    housing_prices_df_raw = pd.read_csv('../../data/raw/train.csv')
    test_housing_prices_df_raw = pd.read_csv('../../data/raw/test.csv')

    train_hdf = housing_prices_df_raw.copy()
    test_hdf = test_housing_prices_df_raw.copy()

    hdf = pd.concat([train_hdf, test_hdf], axis = 0, sort = False)
return hdf


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

    hdf[['Fence']] = hdf[['Fence']].replace({'MnPrv': 'HasFence', 'GdWo': 'HasFence', 
                                            'GdPrv': 'HasFence', 'MnWw': 'HasFence', 
                                            np.nan: 'NoFence'})

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


    hdf.drop(columns = ['Utilities', 'Street', 'PoolQC'], inplace = True)
    hdf.drop(columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], inplace = True)
    hdf.drop(columns = ['FullBath', 'BsmtFullBath', 'HalfBath','BsmtHalfBath'], inplace = True)
    hdf.drop(columns = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF'], inplace = True)
    hdf.drop(columns = 'MiscFeature', inplace = True)
    hdf.drop(columns = 'TotRmsAbvGrd', inplace = True)


    ord_feat_num = set(ord_feat_num).union(set(['TotalBath', 'OtherRoomsAbvGrd', 'Remodeled'])) - \
    set(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',' KitchenAbvGr', 'GarageCars'])

    ord_feat_cat = set(ord_feat_cat) - set(['PoolQC'])

    nom_feat = set(nom_feat) - set(['Utilities','MiscFeature','Street'])

    cont_feat = set(cont_feat).union(set(['TotalPorchAreasSF'])) - set(['Utilities', 'Street', 'PoolQC'] +\
                                                                                                              ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'] +\
                                                                                                              ['FullBath', 'BsmtFullBath', 'HalfBath','BsmtHalfBath'] +\
                                                                                                              ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF'] +\
                                                                                                              ['MiscFeature', 'TotRmsAbvGrd'])

return hdf, ord_feat_num, ord_feat_cat, nom_feat, cont_feat

def Train_test_random_forest(hdf, ord_feat_num, ord_feat_cat, nom_feat, cont_feat):
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
    X_train.to_csv('X_train_forest')
    y_train.to_csv('y_train_forest')
    X_test.to_csv('X_test_forest')
