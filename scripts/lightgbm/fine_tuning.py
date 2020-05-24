import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import os
import shutil
import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper, CategoricalImputer
import re
from datetime import datetime
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# BASE_DIR = os.getcwd()
# BASE_DIR = os.path.dirname(BASE_DIR)
# DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# DATA_EXT_DIR_PATH = os.path.join(DATA_DIR, "init_price_cleaned.csv")

data_ext = pd.read_csv("init_price_cleaned.csv")

print("Chargé la data")

# changer data type: str + float mixture => int => str
data_ext["portes"][data_ext["portes"].notnull()] = data_ext["portes"][data_ext["portes"].notnull()]\
                                                    .astype(int, inplace=True)\
                                                    .astype(str, inplace=True)

data_ext["portes"] = np.where(
    ((data_ext["portes"].isnull()) | (data_ext["portes"] == '0') | (data_ext["portes"] == '6')) \
         & (data_ext["portes_scrap"].notnull()),
    data_ext["portes_scrap"],
    data_ext["portes"]
)

class DateOnlineEncoder(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        d_t = [datetime.strptime(c[0], "%d/%m/%Y à %Hh%M") for c in X.values]
        return np.array(d_t)
    
class AgeFeature(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        #d_t = [datetime.strptime(c, "%d/%m/%Y à %Hh%M") for c in X.iloc[:,0].values]
        age = [np.abs(a.year - b) for (a, b) in zip(X.iloc[:,0].values, X.iloc[:,1].values)]
        return np.array(age)

class MileageClean(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        #assert isinstance(X, pd.DataFrame)
        return np.array([float(m[0].rstrip('km').strip()) for m in np.array(X.values).reshape(-1, 1)]).reshape(-1,1)
    
class DesciptionClean(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        def parser():
            regex_pattern = r"modele:\s*(?P<modele>.*?(?=,)),\sversion:\s*(?P<version>.*?(?=,)),\spuissance_fiscale:\s*(?P<puissance_fiscale>.*?(?=,)),\sportes:\s*(?P<portes>.*?(?=,)),\soptions:\s*(?P<Descriptions>.*?(?=,)),\scouleur:\s(?P<couleur>.*$)"
            regex_cyclindre = "\d+[\.,]\d+"
            regex_cv = "\s+\d{1,3}\s?"
            #version = []
            #puissance_fiscale = []
            #portes = []
            #options = []
            #couleur = []
            for i in range(X.shape[0]):
                match = re.search(regex_pattern, X.values[i][0])
                version = match.group(2)
                if str(version) == 'ii allurehdifap2.0150cv':
                    version = 'ii allurehdifap 2.0 150cv'
                version = re.sub("\d+[\.,]\d+km", "", version)
                version = re.sub("(159.226|76.538|87.480|71.000)", "", version)
                cl = re.findall(regex_cyclindre, version)
                version = re.sub(regex_cyclindre, "", version)
                version = re.sub("\d+p", "", version)
                cv = re.findall(regex_cv, version)
                if len(cl) == 0:
                    cl = np.nan
                else:
                    cl = float(cl[0].strip().replace(",", "."))
                if len(cv) == 0:
                    cv = np.nan
                else:
                    cv = int(float(cv[0].strip()))
                #version.append(match.group(2))
                #puissance_fiscale.append(match.group(3))
                #portes.append(match.group(4))
                #options.append(match.group(5))
                #couleur.append(match.group(6))
                yield [cl, cv, pd.to_numeric(match.group(3)), pd.to_numeric(match.group(4)), str(match.group(6)).lower()]

        return pd.DataFrame.from_records(list(parser()))

class CylindreFeature(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        regex_cyclindre = "\d+[\.,]\d+"
        regex_cv = "\s+\d{1,3}\s?"
        def parser():
            for version in X.values:
                version = version[0]
                if str(version) == 'ii allurehdifap2.0150cv':
                        version = 'ii allurehdifap 2.0 150cv'
                version = re.sub("\d+[\.,]\d+km", "", version)
                version = re.sub("(159.226|76.538|87.480|71.000)", "", version)
                cl = re.findall(regex_cyclindre, version)
                version = re.sub(regex_cyclindre, "", version)
                version = re.sub("\d+p", "", version)
                cv = re.findall(regex_cv, version)
                if len(cl) == 0:
                    cl = np.nan
                else:
                    cl = float(cl[0].strip().replace(",", "."))
                if len(cv) == 0:
                    cv = np.nan
                else:
                    cv = int(float(cv[0].strip()))
                yield [cl]
        return pd.DataFrame.from_records(list(parser()))

class CVFeature(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        regex_cyclindre = "\d+[\.,]\d+"
        regex_cv = "\s+\d{1,3}\s?"
        def parser():
            for version in X.values:
                version = version[0]
                if str(version) == 'ii allurehdifap2.0150cv':
                        version = 'ii allurehdifap 2.0 150cv'
                version = re.sub("\d+[\.,]\d+km", "", version)
                version = re.sub("(159.226|76.538|87.480|71.000)", "", version)
                cl = re.findall(regex_cyclindre, version)
                version = re.sub(regex_cyclindre, "", version)
                version = re.sub("\d+p", "", version)
                cv = re.findall(regex_cv, version)
                if len(cl) == 0:
                    cl = np.nan
                else:
                    cl = float(cl[0].strip().replace(",", "."))
                if len(cv) == 0:
                    cv = np.nan
                else:
                    cv = int(float(cv[0].strip()))
                yield [cv]
        return pd.DataFrame.from_records(list(parser()))

class LinearRegressorImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        #X.iloc[:,0].values, X.iloc[:,1].values
        assert isinstance(X, pd.DataFrame)
        missing_data_price = X.loc[X.iloc[:,0].isnull().values, "Price"].values.reshape(-1,1)
        missing_data_index = [i[0] for i in X.loc[X.iloc[:,0].isnull().values, "Price"].index.values.reshape(-1,1)]
        dt = X.loc[X.iloc[:,0].notnull(), :]
        m = LinearRegression()
        m.fit(X = dt.iloc[:,1].values.reshape(-1,1), y = dt.iloc[:, 0].values.reshape(-1,1))
        missing_data_pred = m.predict(missing_data_price)
        
        X.iloc[missing_data_index, 0] = missing_data_pred.reshape(1, -1)[0]
        return X.iloc[:,0].values

class DummiesCategory(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(pd.get_dummies(X, prefix=self.columns).values)
        return pd.get_dummies(X, prefix=self.columns).values
    
    
class CategoryType(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #assert isinstance(X, pd.DataFrame)
        return X.astype("object")

class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

print("preprocessing")

preprocessing_mapper = DataFrameMapper([
    (["Online"], DateOnlineEncoder()),
    ("Make", [CategoricalImputer(), CategoryType()]),
    ("Model", [CategoryType(), CategoricalImputer()]),
    ("Model_year", CategoricalImputer()),
    ("Mileage", [MileageClean(), Imputer(strategy='mean')]),
    ("Fuel", [CategoryType(), CategoricalImputer()]),
    ("Gearbox", [CategoryType(), CategoricalImputer()]),
], input_df=True, df_out=True, default=None)

data_preprocessing = preprocessing_mapper.fit_transform(data_ext)

print('features extra')
features_mapper = DataFrameMapper([
    (["Online", "Model_year"], AgeFeature(), {'alias': 'age'}),
    (["Model_year"], None),
    (["version"], CylindreFeature(), {'alias': 'cylindre'}),
    (["version"], CVFeature(), {'alias': 'cv'}),
], input_df=True, df_out=True, default=None)

data_extra_features = features_mapper.fit_transform(data_preprocessing)

print("imputation features")
imputer_extra_mapper = DataFrameMapper([
    ("portes", CategoricalImputer()),
    ("couleur", CategoricalImputer()),
    (["cylindre", "Price"], LinearRegressorImputer(), {'alias': 'cylindre'}),
    (["cv", "Price"], LinearRegressorImputer(), {'alias': 'cv'}),
    ("Price", None)
], input_df=True, df_out=True, default=None)

data_imp = imputer_extra_mapper.fit_transform(data_extra_features)

print("category features")
cat_features_mapper = DataFrameMapper([
    ("couleur", LabelEncoder())
], input_df=True, df_out=True, default=None)

data = cat_features_mapper.fit_transform(data_imp)

print("One hot encoding")

data = pd.get_dummies(data, columns=["Make", "Model", "Fuel", "Gearbox", "portes"])

drop_cols = ["options", "puiss_scrap", "portes_scrap", "Price/Starting Price"]
data.drop(drop_cols, axis=1, inplace=True)

data = data[data['starting_price'].notnull()]


target = np.log(data["Price"].values)
data = data.drop(["Price"], axis=1)

print("data scale")

data_scale = StandardScaler().fit_transform(data)

pca = PCA(0.9).fit(data_scale)

print("pca data")

data_pca = pca.transform(data_scale)

print("split data training et testing")
X_train, X_test, y_train, y_test = train_test_split(data_pca, target, test_size=0.2, random_state=42)

X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2)

param_test ={'num_leaves': sp_randint(6, 50),
             'objective' : ['regression'],
             'boosting_type' : ['gbdt', 'dart', 'fr'],
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'mape', 
            "eval_set" : [(X_val,y_val)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}
def MAPE(y_true, y_pred):
    # assuré que les parameters entrés sont 1D array
    y_true = np.array(y_true).reshape(1, -1)[0]
    y_pred = np.array(y_pred).reshape(1, -1)[0]
    return np.mean(np.abs(y_true - y_pred) / y_true)

mape_scorer = make_scorer(MAPE, greater_is_better=False)

lgb_model = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True, metric='mape', n_jobs=4, n_estimators=5000)
print("Starting fine tuning hyperparameter")
gs = RandomizedSearchCV(
    estimator=lgb_model, param_distributions=param_test, 
    n_iter=100,
    scoring=mape_scorer,
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)

gs.fit(X_train_, y_train_, **fit_params)

print("End fine tuning hyperparameter")

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))