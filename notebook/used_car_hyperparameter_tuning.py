import warnings
warnings.filterwarnings('ignore')

import shutil

import pandas as pd
import numpy as np

import os
import re

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV

NOTEBOOK_DIR = os.getcwd()
BASE_DIR = os.path.dirname(NOTEBOOK_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR_PATH = os.path.join(DATA_DIR, "Data_cars.csv")
DATA_ZIP_DIR_PATH = os.path.join(DATA_DIR, "Data_cars.csv.zip")

def online_clean(df):
    datetime_format = "%d/%m/%Y à %Hh%M"
    df.Online = [datetime.strptime(d, datetime_format) for d in df.Online.values]
    return df

def mileage_clean(df):
    df.Mileage = [float(m.split(' ')[0]) for m in df.Mileage.values]
    return df

def descriptions_clean(df):
    regex_pattern = r"modele:\s*(?P<modele>.*?(?=,)),\sversion:\s*(?P<version>.*?(?=,)),\spuissance_fiscale:\s*(?P<puissance_fiscale>.*?(?=,)),\sportes:\s*(?P<portes>.*?(?=,)),\soptions:\s*(?P<Descriptions>.*?(?=,)),\scouleur:\s(?P<couleur>.*$)"
    version = []
    puissance_fiscale = []
    portes = []
    options = []
    couleur = []
    for i in range(df.shape[0]):
        match = re.search(regex_pattern, df.Description[i])
        version.append(match.group(2))
        puissance_fiscale.append(match.group(3))
        portes.append(match.group(4))
        options.append(match.group(5))
        couleur.append(match.group(6))
    df["version"] = list(map(str.lower, version))
    df["puissance_fiscale"] = puissance_fiscale
    df["portes"] = portes
    df["options"] = options
    df["couleur"] = list(map(str.lower, couleur))
    del df["Description"]
    return df

def offre_duplication_clean(df):
    fix_col = list(df.columns)
    fix_col.remove("Online")
    fix_col.remove("Price")
    duplicateRowsDF = df[df.sort_values(by="Online").duplicated(fix_col, keep='last')]
    row_maintain = list(set(df.index) - set(list(duplicateRowsDF.index)))
    df = df.loc[row_maintain, :]
    return df

def cylindre_cv_extra(df):
    regex_cyclindre = "\d+[\.,]\d+"
    regex_cv = "\s+\d{1,3}\s?"
    cylindre = []
    cheveaux = []
    for i in range(df.version.shape[0]):
        if df.version[i] == 'ii allurehdifap2.0150cv':
            df.version[i] = 'ii allurehdifap 2.0 150cv'
        #print(i, data_car_preprocessed.version[i])
        text = df.version[i]
        # supprimer les nombres du kilogmetrage dans le text
        text = re.sub("\d+[\.,]\d+km", "", text)
        text = re.sub("(159.226|76.538|87.480|71.000)", "", text)
        cl = re.findall(regex_cyclindre, text)
        text = re.sub(regex_cyclindre, "", text)
        # supprimer les nombres du porte dans text
        text = re.sub("\d+p", "", text)
        cv = re.findall(regex_cv, text)
        if len(cl) == 0:
            #print("None")
            cylindre.append(np.nan)
        else:
            #print("More 2")
            cylindre.append(float(cl[0].strip().replace(",", ".")))

        if len(cv) == 0:
            cheveaux.append(np.nan)
        else:
            cheveaux.append(int(cv[0].strip()))
        #cylindre.append(re.findall(regex, data_car_preprocessed.version[i])[0])
    # print(len(cv), len(cylindre), df.shape)
    df["cylindre"] = cylindre
    df["cv"] = cheveaux
    return df

def price_log_transformation(df):
    df["log_price"] = np.log(df.Price.values)
    #del df["Price"]
    return df

def car_age(df):
    df["age"] = [int(df.loc[i,"Online"].year) - int(df.loc[i, "Model_year"]) for i in range(df.shape[0])]
    return df

def categorical_variables(df):
    del df["options"]
    categorical_v = ["Make", "Model", "Fuel", "Gearbox", "couleur"]
    df[categorical_v] = df[categorical_v].apply(LabelEncoder().fit_transform)
    return df

def preprocessing(data, preprocessors=None):
    df = data.copy(deep=True)
    if preprocessors is not None:
        for preprocessor in preprocessors:
            df = preprocessor(df)
        return df
    else:
        return df

def MAPE(y_true, y_pred):
    y_true = np.array(y_true).reshape(1, -1)[0]
    y_pred = np.array(y_pred).reshape(1, -1)[0]
    return np.mean(np.abs(y_true - y_pred) / y_true)

if not os.path.exists(DATA_DIR_PATH):
    shutil.unpack_archive(DATA_ZIP_DIR_PATH, DATA_DIR, "zip")
    print("[INFO] Archive file unpacked successfully.")

data_car = pd.read_csv(DATA_DIR_PATH)

data = preprocessing(data_car, [online_clean, \
                                mileage_clean, \
                                descriptions_clean, \
                                cylindre_cv_extra, \
                                car_age,
                                price_log_transformation, \
                                categorical_variables])

data[["puissance_fiscale", "portes"]] = data[["puissance_fiscale", "portes"]].apply(pd.to_numeric)

missing_v_feature = ["cylindre", "cv"]
for f in missing_v_feature:
    missing_data_price = data.loc[data[f].isnull().values, "Price"].values.reshape(-1,1)
    missing_data_index = data.loc[data[f].isnull().values, "Price"].index.values
    dt = data.loc[data[f].notnull(), ["Price", f]]

    m = LinearRegression()
    m.fit(X = dt["Price"].values.reshape(-1,1), y = dt[f].values.reshape(-1,1))
    missing_data_pred = m.predict(missing_data_price)
    data[f][missing_data_index] = missing_data_pred.reshape(1, -1)[0]

port_imp = Imputer(missing_values=np.nan, strategy='mean')
data["portes"] = port_imp.fit_transform(data["portes"].values.reshape(-1,1)).reshape(1, -1)[0]

features = ['Make', 'Model', 'Model_year', 'Mileage', 'Fuel', 'Gearbox', 'puissance_fiscale', 'portes', 'couleur', 'age', 'cylindre', 'cv']
target = ["log_price"]


target = data[target]
data = data[features]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rfr = RandomForestRegressor()

grid_search = GridSearchCV(estimator = rfr, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring=mape_scorer)

grid_search.fit(X_train, y_train.values)

print(grid_search.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    mape = MAPE(predictions, test_labels)
    accuracy = 100 - mape*100
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test.values)