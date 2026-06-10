import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, cross_val_score,GroupKFold, cross_validate, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, balanced_accuracy_score, accuracy_score




#df = pd.read_csv("final_dataframe.csv")
#df = pd.read_csv("final_dataframe_byslice.csv")
df = pd.read_csv("final_dataframe_byslice_FLU.csv")
df['gel_concen'] = df['concentration_corrp']
#df['gel_type'] = df['type_asmmean.'].map({'shg': 0, 'flu': 1})

#remove unnecessary columns
df = df[df['slice']%5==0].reset_index(drop=True)  # reset after filtering
groups = df["image_name"].reset_index(drop=True)
df = df.drop(['image_name','slice','z_depth_HistLEN','concentration_corrp','TotalImageArea','distance3d_indnc','neighbor3d_entro','bin_num3d_denth','roi_senth','short_image_name','type_autoc'],axis=1)
#reduce slices and sampling







print(len(df))

#print(df.columns.values)

#print(df.columns.values)


X = df.drop(columns=['gel_concen'])
y = df['gel_concen']

# --- CV evaluation ---

logo = LeaveOneGroupOut()

for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    print(f"Fold {i+1}: train classes={sorted(y.iloc[train_idx].unique())}, "
          f"test classes={sorted(y.iloc[test_idx].unique())}, "
          f"test group={groups.iloc[test_idx].unique()}")
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
bal_acc_scorer = make_scorer(balanced_accuracy_score)
acc_scorer = make_scorer(accuracy_score)
results = cross_validate(rf_cv, X, y, cv=logo, groups=groups,
                         scoring={"acc": acc_scorer,
                                          "bal_acc": bal_acc_scorer})
#print(f"Accuracy:          {results['test_acc'].mean():.3f} ± {results['test_acc'].std():.3f}")
#print(f"Balanced Accuracy: {results['test_bal_acc'].mean():.3f} ± {results['test_bal_acc'].std():.3f}")

# --- Feature importance (separate model, fit on all data for ranking only) ---
rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_importance.fit(X, y)
result = permutation_importance(rf_importance, X, y,
                                n_repeats=20, random_state=42, n_jobs=2)
importance = pd.Series(result.importances_mean, index=X.columns)
importance = importance.sort_values(ascending=False)

fig, ax = plt.subplots()
importance.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_xlabel("Feature")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

# --- Reduced feature CV ---
top_features = importance.head(10).index.tolist()
print(f"Top features: {top_features}")
X_reduced = X[top_features]

rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
bal_acc_scorer = make_scorer(balanced_accuracy_score)
acc_scorer = make_scorer(accuracy_score)
results_reduced = cross_validate(rf_reduced, X_reduced, y, cv=logo, groups=groups,
                                 scoring={"acc": acc_scorer,
                                          "bal_acc": bal_acc_scorer})
print(f"Reduced Accuracy:          {results_reduced['test_acc'].mean():.3f} ± {results_reduced['test_acc'].std():.3f}")
print(f"Reduced Balanced Accuracy: {results_reduced['test_bal_acc'].mean():.3f} ± {results_reduced['test_bal_acc'].std():.3f}")