import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve



auc_scores_test = []
auc_scores_train = []
auc_scores_val = []
folds = [0, 1, 2, 3, 4]
save_path_test = '1_dataset/'
save_path_val = '1_dataset/'
for fold in folds:
    train_file = pd.read_csv(f"1_dataset/train_fold{fold}.csv")
    val_file = pd.read_csv(f"1_dataset/val_fold{fold}.csv")
    test_file = pd.read_csv('1_dataset/test.csv')
    X_tr = train_file.iloc[:, 2:6]
    y_tr = train_file['label']
    X_val = val_file.iloc[:, 2:6]
    y_val = val_file['label']
    X_te = test_file.iloc[:, 2:6]
    y_te = test_file['label']


    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    model1 = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')#isotonic
    model1.fit(X_val, y_val)


    train_proba = model.predict_proba(X_tr)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_te)[:, 1]


    auc_TRAIN = roc_auc_score(y_tr, train_proba)
    auc_VAL = roc_auc_score(y_val, val_proba)
    auc_TEST = roc_auc_score(y_te, test_proba)

    if auc_TRAIN < 0.5:
        auc_TRAIN = 1 - auc_TRAIN
        train_proba = 1 - train_proba
    if auc_VAL < 0.5:
        auc_VAL = 1 - auc_VAL
        val_proba = 1 - val_proba
    if auc_TEST < 0.5:
        auc_TEST = 1 - auc_TEST
        test_proba = 1 - test_proba

    results_df_test = pd.DataFrame({
        'label': y_te,
        'Predicted': test_proba})
    results_df_val = pd.DataFrame({
        'label': y_val,
        'Predicted': val_proba})
    results_df_test.to_csv(save_path_test + 'DT_Logistic_val_fold_' + str(fold) + '.csv', index=False)
    results_df_val.to_csv(save_path_val + 'DT_Logistic_val_fold_' + str(fold) + '.csv', index=False)

    auc_scores_train.append(auc_TRAIN)
    auc_scores_val.append(auc_VAL)
    auc_scores_test.append(auc_TEST)


    fold += 1


print('AUC train for each fold:', auc_scores_train)
mean_auc_train = np.mean(auc_scores_train)
std_auc_train = np.std(auc_scores_train)
print('AUC val for each fold:', auc_scores_val)
mean_auc_val = np.mean(auc_scores_val)
std_auc_val = np.std(auc_scores_val)
print('AUC test for each fold:', auc_scores_test)
mean_auc_test = np.mean(auc_scores_test)
std_auc_test = np.std(auc_scores_test)


z_value = 1.96
se_train = std_auc_train / np.sqrt(len(auc_scores_train))
ci_train = (mean_auc_train - z_value * se_train, mean_auc_train + z_value * se_train)

se_val = std_auc_val / np.sqrt(len(auc_scores_val))
ci_val = (mean_auc_val - z_value * se_val, mean_auc_val + z_value * se_val)

se_test = std_auc_test / np.sqrt(len(auc_scores_test))
ci_test = (mean_auc_test - z_value * se_test, mean_auc_test + z_value * se_test)
print(f'Average train AUC: {mean_auc_train}, Standard Deviation: {std_auc_train}, 95% CI: {ci_train}')
print(f'Average val AUC: {mean_auc_val}, Standard Deviation: {std_auc_val}, 95% CI: {ci_val}')
print(f'Average test AUC: {mean_auc_test}, Standard Deviation: {std_auc_test}, 95% CI: {ci_test}')




