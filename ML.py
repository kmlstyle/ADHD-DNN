# Code for the manuscript "Predicting childhood and adolescent attention-deficit/hyperactivity disorder onset: a nationwide deep learning approach"
# Author: Miguel Garcia-Argibay
# Orebro University, Sweden
# June, 2022

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, scale, StandardScaler, normalize
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix, roc_curve, auc, roc_auc_score, cohen_kappa_score, precision_recall_curve
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import imblearn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

def linspace(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1)).tolist()

dat = pd.read_csv("Z:/ML/data.csv", delimiter=",")

preds = ['sex', 'head_circ', 'small_age', 'depre', 'anx', 'asd', 'eatingdis',
       'sleepdisorder', 'alrhin_conj', 'adermatitis', 'spch_learng',
       'motortic', 'asthma_rel', 'eatingdis_rel', 'anx_rel', 'nr_fails',
       'dep_rel', 'sud_rel', 'alc_rel', 'adhd_rel', 'crime_rel',
       'crime_i']

X = dat[preds]
y = pd.DataFrame(dat['adhd'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 222, stratify=y)
oversample = BorderlineSMOTE(random_state=888, n_jobs=6, kind='borderline-1')
X_train, y_train = oversample.fit_resample(X_train, y_train)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)

x_train_scaled2 = pd.DataFrame(X_train2, columns = X_train.columns)
x_test_scaled2 = pd.DataFrame(X_test2, columns = X_test.columns)


#rf

rfc=RandomForestClassifier(random_state=42, n_jobs=6)

param_grid = { 
    'n_estimators': [20,50,70,100,200,300,400],
    'max_depth' : linspace(4, 50, step=2),
     'min_samples_split': [2,4,6,8,10], 
     'min_samples_leaf': [1,2,4]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10, scoring = 'roc_auc')
CV_rfc.fit(x_train_scaled2, np.ravel(y_train))

CV_rfc.best_params_

rfc1=RandomForestClassifier(random_state=42, n_estimators= 400, max_depth=50, min_samples_split=2, min_samples_leaf=1)

rfc1.fit(x_train_scaled2, np.ravel(y_train))
y_pred = rfc1.predict(x_test_scaled2)


# For final table
print("Balanced Accuracy :", balanced_accuracy_score(y_test, y_pred))
print("training AUC :", roc_auc_score(y_train, (rfc1.predict_proba(x_train_scaled2)[:,1])))
print("testing AUC :", roc_auc_score(y_test, (rfc1.predict_proba(x_test_scaled2)[:,1])))
precision, recall, thresholds = precision_recall_curve(y_test, rfc1.predict_proba(x_test_scaled2)[:,1])
area = auc(recall, precision)
print("Area Under PR Curve(AP): %0.4f" % area)

#xgb

param_grid = { 
    'n_estimators': [20, 50, 70, 100, 200],
    'reg_lambda': linspace(0, 10, step=0.2),
    'gamma': linspace(0, 10, step=0.2),
    'learning_rate': linspace(0.05, 0.2, step=0.2),
    'max_depth' : linspace(1, 20, step=2),
    'scale_pos_weight' : [None, 1, 2, 3]
}

CV_clf = GridSearchCV(estimator=xgb.XGBClassifier(seed=4, objective= 'binary:logistic'),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv= 10,
                      n_jobs=6,
                      verbose=0)
CV_clf.fit(x_train_scaled2, np.ravel(y_train))

CV_clf.best_params_

clf = xgb.XGBClassifier(gamma = 0.1, objective= 'binary:logistic', n_estimators=200,
                       learning_rate=0.2, max_depth=20, reg_lambda=0.1)
clf.fit(x_train_scaled2, 
        np.ravel(y_train), 
        early_stopping_rounds=30, 
        eval_metric="auc", 
        eval_set=[(x_test_scaled2, np.ravel(y_test))], verbose=False)
		
y_pred = clf.predict(x_test_scaled2)
print("training AUC :", roc_auc_score(y_train, (clf.predict_proba(x_train_scaled2)[:,1])))
print("testing AUC :", roc_auc_score(y_test, (clf.predict_proba(x_test_scaled2)[:,1])))
print("Balanced Accuracy :", balanced_accuracy_score(y_test, y_pred))
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(x_test_scaled2)[:,1])
area = auc(recall, precision)
print("Area Under PR Curve(AP): %0.4f" % area)


from xgboost import plot_importance
from matplotlib import pyplot


sorted_idx = clf.feature_importances_.argsort()
plt.figure(figsize=(7, 6), dpi=100)
plt.barh(X_train.columns[sorted_idx], clf.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")
plt.show()

impor = pd.DataFrame({'Feature':X_train.columns[sorted_idx], 'Importance':clf.feature_importances_[sorted_idx]})
impor.sort_values(by=['Importance'], ascending=False)



# GB
clf=GradientBoostingClassifier(random_state=0)

param_grid = { 
    'n_estimators': [20, 50, 70, 100, 200, 300, 400],
    'learning_rate': linspace(0.05, 0.2, step=0.2),
    'max_depth' : linspace(4, 50, step=2),
    'min_samples_split': [2,4,6,8,10], 
    'min_samples_leaf': [1,2,4],
    'criterion': ['gini', 'entropy', 'log_loss']
}

CV_clf = GridSearchCV(estimator=clf,
                      param_grid=param_grid,
                      cv= 10,
                      scoring = 'roc_auc',
                      verbose=0)


CV_clf.fit(x_train_scaled2, np.ravel(y_train))
CV_clf.best_params_
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=8, random_state=0).fit(x_train_scaled2, np.ravel(y_train))
clf.score(x_test_scaled2, np.ravel(y_test))

sorted_idx = GBoost.feature_importances_.argsort()
plt.figure(figsize=(7, 6), dpi=100)
plt.barh(X_train.columns[sorted_idx], GBoost.feature_importances_[sorted_idx])
plt.xlabel("Gradient boosting Feature Importance")
plt.show()
impor = pd.DataFrame({'Feature':X_train.columns[sorted_idx], 'Importance':GBoost.feature_importances_[sorted_idx]})
impor.sort_values(by=['Importance'], ascending=False)

y_pred = clf.predict(x_test_scaled2)
print("training AUC :", roc_auc_score(y_train, (clf.predict_proba(x_train_scaled2)[:,1])))
print("testing AUC :", roc_auc_score(y_test, (clf.predict_proba(x_test_scaled2)[:,1])))
print("Kappa score testing :", cohen_kappa_score(y_test, y_pred))
print("Balanced Accuracy :", balanced_accuracy_score(y_test, y_pred))
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(x_test_scaled2)[:,1])
area = auc(recall, precision)
print("Area Under PR Curve(AP): %0.4f" % area)

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(25, 5), dpi=500)

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Accuracy")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training accuracy"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation accuracy"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Performance of the model")

    return plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

estimator2 = xgb.XGBClassifier(gamma = 0.25, objective= 'binary:logistic', n_estimators=100,
                       learning_rate=0.1, max_depth=5, reg_lambda=0)
sns.set_theme(style="ticks", font_scale=1.5)
plt.figure(figsize=(7, 5), dpi=100)
plt.xlabel('Training samples')
plt.ylabel('Accuracy') 
plt.title('XGBoost')
plot_learning_curves(estimator2, x_train_scaled2, y_train, cv=10)


# PLG
l2reg = LogisticRegression(random_state=0, n_jobs=6, penalty="elasticnet",l1_ratio=1, C=0.003, solver='saga').fit(x_train_scaled2,np.ravel(y_train))
l2reg = LogisticRegression(random_state=0, n_jobs=6, penalty="l1", C=0.005, solver='saga').fit(x_train_scaled2,np.ravel(y_train))


param_grid = { 
    'classifier__C' : [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    'classifier__penalty' : ['l1', 'l2', 'elasticnet']
}

param_grid = { 
    'classifier__C' : [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    'classifier__penalty' : ['elasticnet'],
    'l1_ratio': [0, 0.2, 0.4, 0.8, 1.0]
}

CV_clf = GridSearchCV(estimator=LogisticRegression(),
                      param_grid=param_grid,
                      cv= 10,
                      scoring = 'roc_auc',
                      verbose=0)


CV_clf.fit(x_train_scaled2, np.ravel(y_train))
CV_clf.best_params_



y_pred = l2reg.predict(x_test_scaled2)
print('Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print("training AUC :", roc_auc_score(y_train, (l2reg.predict_proba(x_train_scaled2)[:,1])))
print("testing AUC :", roc_auc_score(y_test, (l2reg.predict_proba(x_test_scaled2)[:,1])))
#print("Kappa score testing :", cohen_kappa_score(y_test, y_pred))
print("Balanced Accuracy :", balanced_accuracy_score(y_test, y_pred))
precision, recall, thresholds = precision_recall_curve(y_test, l2reg.predict_proba(x_test_scaled2)[:,1])
area = auc(recall, precision)
print("Area Under PR Curve(AP): %0.4f" % area)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()



param_grid = { 
    'var_smoothing' : [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}

CV_clf = GridSearchCV(estimator=gnb,
                      param_grid=param_grid,
                      cv= 10,
                      scoring = 'roc_auc',
                      verbose=0)

CV_clf.fit(x_train_scaled2, np.ravel(y_train))
CV_clf.best_params_

gnb = GaussianNB(var_smoothing=1e-5)
gnb.fit(np.array(x_train_scaled2), np.ravel(y_train))
preds2 = gnb.predict(x_test_scaled2)

print("Train AUC :", roc_auc_score(y_train, (gnb.predict_proba(x_train_scaled2)[:,1])))
print("testing AUC :", roc_auc_score(y_test, (gnb.predict_proba(x_test_scaled2)[:,1])))
print("Balanced Accuracy :", balanced_accuracy_score(y_test, preds2))
precision, recall, thresholds = precision_recall_curve(y_test, gnb.predict_proba(x_test_scaled2)[:,1])
area = auc(recall, precision)
print("Area Under PR Curve(AP): %0.4f" % area)



# Ensemble without DNN

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# Ensemble model



xgb = xgb.XGBClassifier(gamma = 0.1, objective= 'binary:logistic', n_estimators=200, learning_rate=0.2, max_depth=20, reg_lambda=0.1, scale_pos_weight=1, n_jobs=6)
gnb = GaussianNB(var_smoothing=1e-5)
gdb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=20, random_state=0, criterion='gini', min_samples_leaf=1, min_samples_split=2).fit(x_train_scaled2, np.ravel(y_train))
clf = LogisticRegression(random_state=0, n_jobs=6, penalty="elasticnet",l1_ratio=1, C=0.003).fit(x_train_scaled2,np.ravel(y_train))


from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[ ('xgb', xgb), ("gnb", gnb), ('l2', clf), ('gdb', gdb)], voting  = "soft", n_jobs=6)
eclf.fit(x_train_scaled2, np.ravel(y_train))



# including SNN

test = pd.DataFrame(gnb.predict_proba(x_test_scaled2)[:,1], columns=['NB'])
xgb = pd.DataFrame(clf.predict_proba(x_test_scaled2)[:,1], columns=['XGBoost'])
gradiant = pd.DataFrame(GBoost.predict_proba(x_test_scaled2)[:,1], columns=['Gradient'])
#loading DNN predictions
NN = np.loadtxt('Z:/test/ML/NNpreds.txt')
NN2= pd.DataFrame(NN, columns=['NN'])

total = test.join(xgb).join(NN2).join(gradiant)

total = total[['NB','XGBoost','NN', 'Gradient']]
total['softclassifier'] = total.mean(axis=1)

print("testing AUC :", roc_auc_score(y_test, np.array(total['softclassifier'])))
precision, recall, thresholds = precision_recall_curve(y_test, np.array(total['softclassifier']))
area = auc(recall, precision)
print("Area Under PR Curve(AP): %0.4f" % area)

predictions = np.array(total['softclassifier'])

for i in range(0, len(predictions)):
    if predictions[i] >= 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print("Balanced Accuracy :", balanced_accuracy_score(y_test, y_pred))
