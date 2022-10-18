# Code for the manuscript "Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach"
# Author: Miguel Garcia-Argibay
# Orebro University, Sweden
# June, 2022

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, scale, StandardScaler, normalize, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, plot_confusion_matrix, roc_curve, auc, roc_auc_score, cohen_kappa_score, precision_recall_curve, classification_report, confusion_matrix
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

import hyperopt
import tensorflow as tf
import keras
from keras.layers import concatenate, Lambda
from hyperopt import fmin, hp
import sys
import keras_tuner
from keras_tuner.tuners import RandomSearch, BayesianOptimization

from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy import stats

from hyperopt import Trials, STATUS_OK, tpe
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras import optimizers

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping,CSVLogger

from numpy.random import seed
seed(155)
from tensorflow.random import set_seed
import shap
import warnings
warnings.filterwarnings('ignore')
set_seed(222)


import scipy.stats
from scipy import stats

# CI for AUC

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


# Data processing


dat = pd.read_csv("Z:/ML/data.csv", delimiter=",")

preds = ['sex', 'head_circ', 'small_age', 'depre', 'anx', 'asd', 'eatingdis',
       'sleepdisorder', 'alrhin_conj', 'adermatitis', 'spch_learng',
       'motortic', 'asthma_rel', 'eatingdis_rel', 'anx_rel',
       'dep_rel', 'sud_rel', 'alc_rel', 'adhd_rel', 'crime_rel', 'nr_fails',
       'crime_i']

X = dat[preds]
y = pd.DataFrame(dat['adhd'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 222, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.18, random_state = 222, stratify=y_train)

oversample = BorderlineSMOTE(random_state=888, n_jobs=6, kind='borderline-1')
X_train, y_train = oversample.fit_resample(X_train, y_train)

scaler = MinMaxScaler().fit(X_train)
X_train2 = scaler.transform(X_train)
X_valid2 = scaler.transform(X_valid)
X_test2 = scaler.transform(X_test)
x_train_scaled2 = pd.DataFrame(X_train2, columns = X_train.columns)
x_test_scaled2 = pd.DataFrame(X_test2, columns = X_test.columns)
x_valid_scaled2 = pd.DataFrame(X_valid2, columns = X_valid.columns)


for i in ['sex', 'small_age', 'depre', 'anx', 'asd', 'eatingdis',
       'sleepdisorder', 'alrhin_conj', 'adermatitis', 'spch_learng',
       'motortic', 'asthma_rel', 'eatingdis_rel', 'anx_rel',
       'dep_rel', 'sud_rel', 'alc_rel', 'adhd_rel', 'crime_rel',
       'crime_i']:
    x_train_scaled2[i] = x_train_scaled2[i].astype(np.int64)



# DNN set up and parameter optimization

input_dim = len(np.array(x_train_scaled2)[1])
output_dim = len(np.array(y_train)[1])
print("input", input_dim, "output dimensions", output_dim)

space = {
    'layer1': hp.quniform('layer1', 5, 20, 1),
    'dropout1' : hp.uniform('dropout1', 0, 0.5),
    'layer2': hp.quniform('layer2', 5, 80, 1),
    'batch_size': hp.quniform('batch_size', 5, 80, 10),
    'optimizer': hp.choice('optimizer', [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD, tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adagrad,
                                         tf.keras.optimizers.Adadelta, tf.keras.optimizers.Adamax, tf.keras.optimizers.Nadam]),
    'lr': hp.uniform('lr', 0, 0.1),
    'l1': hp.choice('lr', [0, 1e-4, 1e-3, 1e-2]),
    'act': hp.choice('act', ['relu', 'selu','LeakyReLU']),
}

def f_nn(params):
    print('Params testing: ', params)
    input = Input(shape=(input_dim,))
    fc = Dense(units=int(params['layer1']), activation=params['act'], kernel_regularizer=regularizers.l1(l1=params['l1']), name ="fc1")(input)
    fc = Dropout(params['dropout1'])(fc)
    fc2 = Dense(units=int(params['layer2']), activation=params['act'], name ="fc2")(fc)
    out = Dense(units=1, activation='sigmoid', name = 'out')(fc2)
    model = Model(input, out)
    opt = params['optimizer'](learning_rate=params['lr'])

    model.compile(loss=['binary_crossentropy'], metrics=tf.metrics.AUC(name='AUC'), optimizer=opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=5)

    History = model.fit(x=np.array(x_train_scaled2), y = np.array(y_train),
                        epochs=80,  # int(params['n_epochs']),
                        batch_size=int(params['batch_size']),
                        shuffle=True, verbose=2,
                        validation_data=(x_valid_scaled2, y_valid),
                        callbacks=[es])

    return {'loss': History.history['val_loss'][-1],
            'train_loss': History.history['loss'][-1],
            'status': STATUS_OK, 'params': params}

trials = Trials()

best = fmin(f_nn, space, algo=tpe.suggest, max_evals=15, trials=trials)

with open("trial_obj.pkl", "wb") as f:
    pickle.dump(trials, f, -1)

# save the search results
f = open("HO_results.log", "w")
for i, tr in enumerate(trials.trials):
    f.write("Trial; " + str(i) + ";" + "train_loss;" + str(tr['result']['train_loss'])
            + ";" + "valid_loss; " + str(tr['result']['loss']) + ";"
            +";" + "Parameters; " + str(tr['result']['params']) + "\n")
f.close()

print("*" * 150)
print(best)
print(hyperopt.space_eval(space, best))
print("*" * 150)



# Bayesian Optimization

import tensorflow as tf
from bayes_opt import BayesianOptimization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')


# Make scorer accuracy
score_acc = make_scorer(accuracy_score)

input_dim = len(np.array(x_train_scaled2)[1])
output_dim = len(np.array(y_train)[1])
print("input", input_dim, "output dimensions", output_dim)


def nn_cl_bo(neurons, neurons2, dropp, activation, optimizer, learning_rate,  batch_size, epochs ):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    optimizerD= {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                 'RMSprop':tf.keras.optimizers.RMSprop(lr=learning_rate), 'Adadelta':tf.keras.optimizers.Adadelta(lr=learning_rate),
                 'Adagrad':tf.keras.optimizers.Adagrad(lr=learning_rate), 'Adamax':tf.keras.optimizers.Adamax(lr=learning_rate),
                 'Nadam':tf.keras.optimizers.Nadam(lr=learning_rate), 'Ftrl':tf.keras.optimizers.Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', 'LeakyReLU' ,'relu']
    neurons = round(neurons)
    neurons2 = round(neurons2)
    activation = activationL[round(activation)]
    batch_size = round(batch_size)
    optimizer = optimizerD[optimizerL[round(optimizer)]]
    epochs = round(epochs)
    def nn_cl_fun():
        nn = Sequential()
        nn.add(Dense(neurons, input_dim=22, activation=activation))
        nn.add(BatchNormalization())
        nn.add(Dense(neurons2, activation=activation))
        nn.add(Dropout(dropp))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])
        return nn
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=12)
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size,
                         verbose=0)
    hist = nn.fit(x_train_scaled2, y_train, validation_data=(X_val, y_val), verbose=1)
    return score
    
params_nn ={
    'neurons': (10, 20),
    'neurons2': (5, 20),
    'dropp': (0.1, 0.3),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.001, 0.01),
    'batch_size':(20, 60),
    'epochs':(30, 100)
}
nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=111)
nn_bo.maximize(init_points=5 n_iter=200)

params_nn_ = nn_bo.max['params']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', LeakyReLU,'relu']
params_nn_['activation'] = activationL[round(params_nn_['activation'])]
params_nn_


#Bayesian optimization - 2

# Optimize the number of layers as well

def build_model2(hp):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(units=hp.Int('units_primera', 5, 20, step=1), activation='relu', kernel_regularizer=regularizers.l1(l1=hp.Float('l11', 0, 0.001, step=0.0001))))
    model.add(Dropout(hp.Float('LDR', 0, 0.1, step=0.01)))
    for i in range(hp.Int('layers', 1, 3)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 4, 30, step=2),
                                        kernel_regularizer=regularizers.l1(l1=hp.Float('l1', 0, 0.001, step=0.0001)),
                                        activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid', 'selu', 'LeakyReLU'])))
    model.add(Dense(1, activation='sigmoid'))
    lr=hp.uniform('lr', 0.0001, 0.1)
    opt = tf.optimizers.Adadelta(learning_rate=lr)
    model.compile(opt, 'binary_crossentropy', metrics=tf.metrics.AUC(name='AUC'))
    return model

class MyTuner(keras_tuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 25, 50, step=5)
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)
		
		
tuner2 = MyTuner(
    build_model2,
    objective = 'val_loss',
    max_trials = 200,
    num_initial_points=5,
    alpha=0.0001,
    beta=2.6,
    directory = 'Z:/test/ML',
    overwrite=False
)


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
tuner2.search(x=np.array(x_train_scaled2), y = np.array(y_train),
              epochs=110, validation_data=(x_valid_scaled2, y_valid),  callbacks=[es],
             use_multiprocessing=True,
             workers=6)
			 

# Get the optimal hyperparameters
best_hps=tuner2.get_best_hyperparameters(num_trials=1)[0]


print(f""" The best model had:
Number of hidden layers: {best_hps.get('layers')}
Neurons first layer {best_hps.get('units_0')} with activation {best_hps.get('act_0')}
Neurons 2 layer {best_hps.get('units_1')} with activation {best_hps.get('act_1')}
Neurons 3 layer {best_hps.get('units_2')} with activation {best_hps.get('act_2')}
l1 regularization:  {best_hps.get('l1')}
Batch {best_hps.get('batch_size')}
LR {best_hps.get('lr')}.
""")




# Best model - HyperOPT!
model = Sequential()
model.add(Input(shape=(input_dim,)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(l1=1e-3)))
model.add(Dropout(0.1565))
model.add(Dense(units = 7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = tf.optimizers.Adadelta(learning_rate=0.006833192465340967)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=tf.metrics.AUC(name='AUC'))

history = model.fit(x=np.array(x_train_scaled2), y = np.array(y_train), 
          validation_data=(x_valid_scaled2, y_valid), 
          batch_size=60, epochs=200, 
          callbacks=[es],
          shuffle=True, verbose=1)
          

predictions = model.predict(x = x_test_scaled2, batch_size=100000, verbose=0)
fpr2, tpr2, _ = roc_curve(y_test, predictions)
roc_auc2 = auc(fpr2, tpr2)
print(roc_auc2)


alpha = .95
y_pred = predictions
y_true=np.ravel(y_test)

auc, auc_cov = delong_roc_variance(
    y_true,
    y_pred)

auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(
    lower_upper_q,
    loc=auc,
    scale=auc_std)

ci[ci > 1] = 1

print('AUC:', auc)
print('95% AUC CI:', ci)



plt.figure(figsize=(7, 9), dpi=100)
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.subplot(212)
plt.title('AUC')
plt.plot(pd.DataFrame(history.history).iloc[:,1], label='Training')
plt.plot(pd.DataFrame(history.history).iloc[:,3], label='Validation')
plt.legend()
plt.show()

model.save('Z:/ML/best', save_format='h5')


from sklearn.metrics import precision_recall_fscore_support
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)
    
    support = class_report_df.loc['support']
    total = support.sum() 
    

    return class_report_df.T
 
    
vals = [0.13,0.20,0.25,0.30,0.34,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
first1 = pandas_classification_report(y_test, predictions)
first1 = first1[['precision', 'recall']]
first2 = {'Sensitivity': first1['recall'][1], 'Specificity': first1['recall'][0], 'PPV': first1['precision'][1], 'NPV': first1['precision'][0]}
first = pd.DataFrame(data=first2, index=[0])
    
for a in vals:  
    predictions = model.predict(x = x_test_scaled2, batch_size=10000, verbose=0)
    for i in range(0, len(predictions)):
        if predictions[i] >= a:
            predictions[i] = 1
        else:
            predictions[i] = 0
    b = pandas_classification_report(y_test, predictions)
    b = b[['precision', 'recall']]
    b['threshod'] = a
    empa = {'threshod':b['threshod'][1], 'Sensitivity': b['recall'][1], 'Specificity': b['recall'][0], 'PPV': b['precision'][1], 'NPV': b['precision'][0]}
    empa2 = pd.DataFrame(data=empa, index=[0])
    first = first.append(empa2)
    
    
sns.set_theme(style="ticks", font_scale=1.5)
plt.figure(figsize=(6, 5), dpi=100)
lw = 2
plt.plot(fpr2, tpr2, color='darkorange', 
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



precision, recall, _ = precision_recall_curve(y_test, predictions)
pos_probs = model.predict(x = x_test_scaled2, batch_size=5000, verbose=0)
no_skill = len(y_test[y_test['adhd']==1]) / len(y_test)
plt.figure(figsize=(6, 5), dpi=100)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill')
precision, recall, _ = precision_recall_curve(y_test, pos_probs)
plt.plot(recall, precision, marker='.', label='DNN')
plt.xlabel('Sensitivity')
plt.ylabel('Positive Predictive Power')
plt.legend()
plt.show()


oo = np.array(shap.sample(x_train_scaled2, 1000))
oo2 = np.array(shap.sample(x_test_scaled2, 800))
explainer = shap.KernelExplainer(model, x_valid_scaled2)
shap_values = explainer.shap_values(x_valid_scaled2.iloc[100:200,:], nsamples=400)
shap.summary_plot(shap_values, pd.DataFrame(x_test_scaled2), plot_type="bar")
mlp_impt = np.mean(abs(shap_values[0]), axis= 0)
#alternative bar plot instead of summary plot
plt.figure()
plt.bar([x for x in range(len(mlp_impt))] ,mlp_impt)
plt.show()

# -----------------------------