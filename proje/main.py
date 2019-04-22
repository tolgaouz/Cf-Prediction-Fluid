import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) # Limiting floating points to 3

train = pd.read_excel(r'Cf_Re1.xls',sheetname='InputData')
test = pd.read_excel(r'Cf_Re1.xls',sheetname='Prediction')

## Plotting traning data
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['x [m]'],train['value'],'g1')
plt.ylabel('x[m]', fontsize=13)
plt.xlabel('value', fontsize=13)
plt.show()

sns.distplot(train['value'] , fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['value'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('value distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['value'], plot=plt)
plt.show()

## MODELING
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Validation function
n_folds = 5


def rmsle_cv(model, type=None):
    rmse = np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv=n_folds,
                                    estimator_type=type))
    return (rmse)

# huber loose makes it robust to outliers
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# pack up models in a list for easy use
models = [GBoost,model_lgb,ENet,lasso,KRR]

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


def build_model():
    model = Sequential()

    ## Input Layer
    model.add(Dense(32, kernel_initializer='normal', input_dim=x_train.shape[1], activation='relu'))

    ## Hidden Layers
    for r in range(3):
        model.add(Dense(32 * 2, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model

# Checkpoint to keep track of best weight matrices
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# Creating a KerasRegressor to use ANN with scikit-learn later
ANNRegressor = KerasRegressor(build_fn=build_model,epochs=100,batch_size=64,validation_split = 0.2)
ANNRegressor.fit(x_train,y_train)
pred = ANNRegressor.predict(x_test)

# load the best model
wights_file = 'Weights-081--0.01038.hdf5' # choose the best checkpoint
ANNRegressor.model.load_weights(wights_file) # load it
ANNRegressor.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# make the prediction
prediction = ANNRegressor.predict(x_test,batch_size=64)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# evaluate the score of ANN
score = rmsle(y_test,prediction)
print("\nANN score: {:.4f}\n".format(score))

# squeezing prediction array so we can plot it
prediction = np.squeeze(prediction)

def prediction_graph(prediction,title,figsize):
    # percent error
    percent_error = np.multiply(np.abs(np.divide(np.subtract(y_test,prediction),y_test)),100)
    score = rmsle(y_test,prediction)
    # plot to show percent error and predicted values
    fig,(ax1,ax3) = plt.subplots(2,1,figsize=figsize)
    plt.title(title)
    color = 'tab:red'
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('values', color=color)
    ln1 = ax1.plot(x_test['x [m]'],prediction,'r--',label='prediction')
    ln2 = ax1.plot(x_test['x [m]'],y_test,'g1',label='test')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('percent error', color=color)  # we already handled the x-label with ax1
    ln3 = ax2.plot(x_test['x [m]'], percent_error, color=color,label='error(in percentage)')
    ax2.tick_params(axis='y', labelcolor=color)
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid()
    font = {'family' : 'normal',
        'size'   : 13}

    plt.rc('font', **font)
    ax3.set_xlabel('true values')
    ax3.set_ylabel('predicted values', color=color)
    ax3.plot(y_test,prediction,'g1')
    xx = np.linspace(*ax3.get_xlim())
    ax3.plot(xx, xx,color='red')
    plt.grid(True)
    plt.show()
    return score

# ANN prediction graph
prediction_graph(prediction,'ANN',(15,15))