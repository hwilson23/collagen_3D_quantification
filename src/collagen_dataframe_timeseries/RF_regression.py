import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, cross_val_score,GroupKFold, cross_validate, LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance




#df = pd.read_csv("final_dataframe.csv")
#df = pd.read_csv("final_dataframe_byslice.csv")
df = pd.read_csv(r"..\final_dataframe_byslice_FLU.csv")

#df['gel_type'] = df['type_asmmean.'].map({'shg': 0, 'flu': 1})
df['gel_concen'] = df['concentration_corrp'].map({'1mgml':1, '2mgml':2, '3mgml':3})


#remove unnecessary columns
df = df[df['slice']%5==0].reset_index(drop=True)  # reset after filtering
groups = df["image_name"].reset_index(drop=True)
df = df.drop(['image_name','slice','z_depth_HistLEN','concentration_corrp','TotalImageArea','distance3d_indnc','neighbor3d_entro','bin_num3d_denth','roi_senth','short_image_name','type_autoc'],axis=1)
#print(df.columns.values)
df = df.select_dtypes(include=['number'])
#print(df.columns.values)


X = df.drop(columns=['gel_concen'])
y = df['gel_concen']

# --- CV evaluation ---
logo = LeaveOneGroupOut()
rf_cv = RandomForestRegressor(n_estimators=100, random_state=42)
results = cross_validate(rf_cv, X, y, cv=logo, groups=groups,
                         scoring={"r2": "r2",
                                  "mse": "neg_mean_squared_error"})

print(f"R²:  {results['test_r2'].mean():.3f} ± {results['test_r2'].std():.3f}")
print(f"MSE: {(-results['test_mse']).mean():.3f} ± {(-results['test_mse']).std():.3f}")

# --- Feature importance (separate model, fit on all data for ranking only) ---
rf_importance = RandomForestRegressor(n_estimators=100, random_state=42)
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

rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
results_reduced = cross_validate(rf_reduced, X_reduced, y, cv=logo, groups=groups,
                                 scoring={"r2": "r2",
                                          "mse": "neg_mean_squared_error"})

print(f"Reduced R²:  {results_reduced['test_r2'].mean():.3f} ± {results_reduced['test_r2'].std():.3f}")
print(f"Reduced MSE: {(-results_reduced['test_mse']).mean():.3f} ± {(-results_reduced['test_mse']).std():.3f}")

"""
gkf = GroupKFold(n_splits=3)
rf_fresh = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf_fresh, X, y, cv=gkf, groups=groups, scoring="r2")
single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)

print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"R² per fold: {scores}")
print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")

result = permutation_importance(
    rf_regressor, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2)
#importance = pd.Series(rf_regressor.feature_importances_, index=X.columns)
importance = pd.Series(result.importances_mean, index=X.columns)
importance = importance.sort_values(ascending=False)

fig, ax = plt.subplots()
importance.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_xlabel("Mean accuracy decrease")
ax.set_ylabel("Feature")
fig.tight_layout()
print("plots displaying")
plt.show()

top_features = importance.head(10).index.tolist()
X_reduced = X[top_features]
rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
results = cross_validate(rf_reduced, X_reduced, y, cv=gkf, groups=groups,
                         scoring={"r2": "r2", "mse": "neg_mean_squared_error"})

print(f"R²:  {results['test_r2'].mean():.3f} ± {results['test_r2'].std():.3f}")
print(f"MSE: {(-results['test_mse']).mean():.3f} ± {(-results['test_mse']).std():.3f}")

"""