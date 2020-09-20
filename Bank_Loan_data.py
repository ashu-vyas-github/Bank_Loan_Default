import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV

from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, log_loss, average_precision_score
from category_encoders import TargetEncoder, WOEEncoder, HashingEncoder, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier



def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



dpi_setting=1200
plt.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 400

data_path = '/media/ashutosh/Computer Vision/Predictive_Maintenance/Bank_Loan_data_Kaggle'
# data_path = 'E:\Predictive_Maintenance\Bank_Loan_data_Kaggle'

data_csv = pd.read_csv(data_path+"//bank_loan_csv.csv")
data_csv = data_csv.drop(['Count','Customer_ID'],axis=1)
data_labels = data_csv['Default_On_Payment']
data_csv = data_csv.drop(['Default_On_Payment'],axis=1)
features_list = list(data_csv.columns)

print(list(data_csv.columns))

# #### Plotting individual feature distributions
# for onefeature in features_list:
#     plot_df = data_csv[onefeature].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100
#     plot_df_data = list(plot_df)
#     plot_df_uniques = list(plot_df.keys())
#     plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
#     plt.ylim(0,100)
#     plt.xlabel(onefeature)
#     plt.ylabel('Percent [%]')
#     plt.bar(plot_df_uniques,height=plot_df_data, width=0.17, linewidth=0, edgecolor='w')
#     plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
#     plt.savefig(data_path+"//plots//Distribuion_{xaxs}.png".format(xaxs=onefeature), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
#     plt.close()



# feat_list = ['Status_Checking_Acc', 'Credit_History', 'Purposre_Credit_Taken', 'Savings_Acc', 'Years_At_Present_Employment', 'Inst_Rt_Income', 'Marital_Status_Gender', 'Other_Debtors_Guarantors', 'Current_Address_Yrs', 'Property', 'Other_Inst_Plans ', 'Housing', 'Num_CC', 'Job', 'Dependents', 'Telephone', 'Foreign_Worker', 'Default_On_Payment']

# for xaxis_feature in feat_list:
#     feat_list.remove(xaxis_feature)
#     for yaxis_feature in feat_list:
#         if xaxis_feature == yaxis_feature:
#             continue
#         else:
#             plot_2d_raw = data_csv.groupby([yaxis_feature,xaxis_feature])[xaxis_feature].count()
#             plot_2d_list = list(data_csv.groupby([yaxis_feature,xaxis_feature])[xaxis_feature].count())
#             y_raw_labels = list(plot_2d_raw.keys().get_level_values(0))
#             y_labels = np.unique(y_raw_labels)
#             x_raw_labels = list(plot_2d_raw.keys().get_level_values(1))
#             x_labels = np.unique(x_raw_labels)
#             norm_fact = 100.0/data_csv.shape[0]
#             if len(plot_2d_list) != len(y_labels)*len(x_labels):
#                 diff_num_elements = len(y_labels)*len(x_labels) - len(plot_2d_list)
#                 for x in range(diff_num_elements):
#                     plot_2d_list.append(np.nan)
#             plot_2d_data = np.reshape(plot_2d_list,(len(y_labels),len(x_labels)))*norm_fact
#             plot_2d_data = np.ma.masked_invalid(plot_2d_data)
#             plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
#             plt.xlabel(xaxis_feature)
#             plt.xticks(ticks=[x for x in range(len(x_labels))],labels=x_labels)
#             plt.ylabel(yaxis_feature)
#             plt.yticks(ticks=[y for y in range(len(y_labels))],labels=y_labels)
#             gcaobj = plt.gca()
#             gcaobj.patch.set(hatch='x', edgecolor='black')
#             plt.imshow(plot_2d_data, cmap='tab20', norm=None, aspect='equal', interpolation=None, alpha=None, vmin=None, vmax=None, origin='lower', extent=None, filternorm=1, filterrad=4.0, resample=None, url=None, data=None)
#             plt.colorbar()
#             plt.savefig(data_path+"//plots//{xaxs}_{yaxs}.png".format(xaxs=xaxis_feature,yaxs=yaxis_feature), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
#             plt.close()


##### Random Forest Classifier

#### Split training and validation data for measuring performance of the model ####
X_train, X_valid, y_train, y_valid = train_test_split(data_csv, data_labels, test_size = 0.2, stratify=data_labels, random_state=42)


features_numerical = ['Duration_in_Months', 'Credit_Amount', 'Inst_Rt_Income', 'Current_Address_Yrs', 'Age', 'Num_CC', 'Dependents']
stdscl = StandardScaler()
X_train_num_ss = stdscl.fit_transform(X_train[features_numerical])
X_valid_num_ss = stdscl.transform(X_valid[features_numerical])

features_categorical = ['Status_Checking_Acc', 'Credit_History', 'Purposre_Credit_Taken', 'Savings_Acc', 'Years_At_Present_Employment', 'Marital_Status_Gender', 'Other_Debtors_Guarantors', 'Property', 'Other_Inst_Plans ', 'Housing', 'Job', 'Telephone', 'Foreign_Worker']

#### One-Hot Encoding ####
features_ohenc = features_categorical
#one_hot_enc = OneHotEncoder(categories='auto', drop='first', sparse=True, handle_unknown='error')
one_hot_enc = OneHotEncoder(cols=features_ohenc, drop_invariant=False, return_df=True, handle_missing='value', handle_unknown='value', use_cat_names=False)

X_train_ohe = one_hot_enc.fit_transform(X_train[features_ohenc])
X_valid_ohe = one_hot_enc.transform(X_valid[features_ohenc])
#### End ####

train_df_all_enc = hstack((X_train_num_ss, X_train_ohe), format='csr')
valid_df_all_enc = hstack((X_valid_num_ss, X_valid_ohe), format='csr')


ml_model = RandomForestClassifier(n_estimators=24, criterion='entropy', max_depth=21, min_samples_leaf=1, min_samples_split=2, max_features='log2', max_leaf_nodes=None, n_jobs=-1, random_state=42, class_weight=None, bootstrap=True, oob_score=True, warm_start=False)
ml_model.fit(train_df_all_enc, y_train)
# y_pred = ml_model.predict(valid_df_all_enc)
# acc_rfc = round(accuracy_score(y_valid,y_pred) * 100, 2)
# print("RFC Accuracy:",acc_rfc)
# auprc_rfc = average_precision_score(y_valid,y_pred)
# print("RFC AUPRC:",auprc_rfc)



# param_grid = {'bootstrap':[True,False],'oob_score':[True,False],'warm_start':[True,False]}

# cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
# gscv = GridSearchCV(ml_model, param_grid=param_grid, scoring='average_precision', n_jobs=-1, refit=True, cv=cv, verbose=0, pre_dispatch='2*n_jobs', return_train_score=False)
# gscv.fit(train_df_all_enc, y_train)
# y_pred = gscv.predict(valid_df_all_enc)
# auprc_rfc = average_precision_score(y_valid,y_pred)
# print("\nRFC AUPRC:",auprc_rfc)
# print(gscv.best_estimator_)
# print(gscv.best_score_)
# print(gscv.best_params_)

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
rfecv = RFECV(ml_model, step=1, min_features_to_select=1, cv=cv, scoring='average_precision', verbose=10, n_jobs=-1)
X_train_new = rfecv.fit_transform(train_df_all_enc, y_train)
print("Optimal features: %d" % rfecv.n_features_)
X_valid_new = rfecv.transform(valid_df_all_enc)
ml_model.fit(X_train_new, y_train)
y_pred = ml_model.predict(X_valid_new)
acc_model = round(accuracy_score(y_valid,y_pred) * 100, 2)
print("Classifier Acc:",acc_model)
auprc_rfc = average_precision_score(y_valid,y_pred)
print("RFC AUPRC:",auprc_rfc)


### Learning Curves
title = "Learning Curves RFC"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(ml_model, title, X_train_new, y_train, axes=None, ylim=(0.7, 1.1), cv=cv, n_jobs=-1)
plt.show()

### Confusion Matrix
plot_confusion_matrix(ml_model, X_valid_new, y_valid, labels=None, sample_weight=None, normalize='true', display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None)
plt.show()



print("\nDone!!!\n")


