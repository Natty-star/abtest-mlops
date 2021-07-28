
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import pandas as pd


data = pd.read_csv('../data/AdSmartABdata.csv')

#Copying the data for Machine learning
data_ML = data.copy()
# Add row id
data_ML['row_id'] = data_ML.index


# Remove non 
data_ML.dropna(inplace=True)


# Remove Date and Payments columns
del data_ML['auction_id'], data_ML['hour'], data_ML['device_make'], data_ML['browser']

# Shuffle the data
data_ML = sklearn.utils.shuffle(data_ML)


#reordering the Columns
data_ML = data_ML[['row_id', 'experiment', 'yes', 'no', 'platform_os', 'date']]
data_ML.head()

X_train, X_test, y_train, y_test = train_test_split(data_ML.loc[:, data_ML.columns != 'yes'],\
                                                    data_ML['yes'], test_size=0.3) 
X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test, test_size=0.33)

# Converting strings to numbers


lb = LabelEncoder()
X_train['experiment'] = lb.fit_transform(X_train['experiment'])
X_test['experiment'] = lb.transform(X_test['experiment'])
X_valid['experiment'] = lb.transform(X_valid['experiment'])

X_train['date'] = lb.fit_transform(X_train['date'])
X_test['date'] = lb.transform(X_test['date'])
X_valid['date'] = lb.transform(X_valid['date'])

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))
    

plt.style.use('ggplot')


def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('User Who have answered Yes')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()
    
   
X_train_refined = X_train.drop(columns=['row_id'], axis=1)
linear_regression = sm.OLS(y_train, X_train_refined)
linear_regression = linear_regression.fit()

X_valid_refined = X_valid.drop(columns=['row_id'], axis=1)
y_preds = linear_regression.predict(X_valid_refined)

X_test_refined = X_test.drop(columns=['row_id'], axis=1)
y_preds_test = linear_regression.predict(X_test_refined)

plot_preds(y_valid, y_preds, 'Linear Regression')
