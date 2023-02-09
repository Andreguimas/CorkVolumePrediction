import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder   
import pickle
dados_biometricos=pd.read_csv('Dados_Biometricos.txt', header=None)

num_attribs = list(dados_biometricos)

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                         ('MMS_scaler', MinMaxScaler(feature_range = (0, 1)))
                               ])

full_pipeline=ColumnTransformer([
     ('num', num_pipeline, num_attribs)
])

X_dados_prepared=full_pipeline.fit_transform(dados_biometricos)

X_test = X_dados_prepared

# load the model from disk
loaded_model = pickle.load(open("C:/Users/andre/Desktop/Mask RCNN/ProjetoCortica/ModeloML - Cópia/Model/finalized_model.sav", 'rb'))

y_pred_grid = loaded_model.predict(X_test)

pred = np.round(y_pred_grid, 3)

f = open("predicted.txt", "w")
f.write(str(pred[0]))
f.close()