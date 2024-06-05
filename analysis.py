import pandas as pd 

#1 Preprocessing and EDA:
#1A display the first few rows of the CHAMPS dataset
data_frame = pd.read_csv('src/CHAMPS.csv')
print(data_frame.head(), '\n')

#1B Get the number of rows and columns
num_rows, num_cols = data_frame.shape
print(f'The CHAMPS dataset has {num_rows} rows and {num_cols} columns.', '\n')

#1C Enumerate the columns of the dataset
print(data_frame.columns.tolist(), '\n')

#D Rename the columns based on the provided mappings
data_frame.rename(columns={'champs_id':'CHAMPS_ID (Mortality)',
                            'dp_013' : 'case_type',
                            'dp_018' : 'underlying_cause',
                            'dp_118' : 'Main_maternal_condition'}, inplace=True)

#E Rename the values in the case_type column.
case_type_mapping = {
    'CH00716': 'Stillbirth',
    'CH01404': 'Death in the first 24 hours',
    'CH01405': 'Early Neonate (1 to 6 days)',
    'CH01406': 'Late Neonate (7 to 27 days)',
    'CH00718': 'Infant (28 days to less than 12 months)',
    'CH00719': 'Child (12 months to less than 60 months)'
}

data_frame['case_type'] = data_frame['case_type'].map(case_type_mapping)

#F Show the proportion of null values in each column.
null_proportion = data_frame.isnull().mean() * 100
print(null_proportion, '\n')

#Descriptive Data analysis
#2A What are the magnitude and proportion of each of the infant underlying cause for child death?
infant_underlying_cause_counts = data_frame['underlying_cause'].value_counts()
infant_underlying_cause_proportion = data_frame['underlying_cause'].value_counts(normalize=True) * 100

print('magnitude of each of the infant underlying cause for child death', infant_underlying_cause_counts, '\n')
print('Proportion of each of the infant underlying cause for child death', infant_underlying_cause_proportion, '\n')

#2B What are the proportion and magnitude of the maternal factors contributing for child death?
maternal_condition_counts = data_frame['Main_maternal_condition'].value_counts()
maternal_condition_proportions = data_frame['Main_maternal_condition'].value_counts(normalize =True) * 100

print('magnitude of the maternal factors contributing for child death', maternal_condition_counts, '\n')
print('proportion of the maternal factors contributing for child death', maternal_condition_proportions, '\n')

#2C What are the proportion of the child death by the case type
case_type_counts = data_frame['case_type'].value_counts()
case_type_proportion = data_frame['case_type'].value_counts(normalize= True) * 100

print('magnitude of the child death by the case type', case_type_counts, '\n')
print('proportion of the child death by the case type', case_type_proportion, '\n')

#3 Correlation analysis:
#Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data_frame['case_type_encoded'] = label_encoder.fit_transform(data_frame['case_type'].astype(str))
data_frame['underlying_cause_encoded'] = label_encoder.fit_transform(data_frame['underlying_cause'].astype(str))
data_frame['maternal_condition_encoded'] = label_encoder.fit_transform(data_frame['Main_maternal_condition'].astype(str))

# Select only numerical columns for correlation analysis
numerical_data_frame = data_frame[['case_type_encoded', 'maternal_condition_encoded']]
#create correlation matrix
correlation_matrix = numerical_data_frame.corr()

#plot the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot= True, cmap = 'coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#4 Feature Engineering
#4B # Plot feature importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
import numpy as np
#4A Select the classification models
# Select the top 3 underlying causes identified from 2(A)
top_causes = data_frame['underlying_cause'].value_counts().index[:3]
data_frame = data_frame[data_frame['underlying_cause'].isin(top_causes)]

# Feature selection
features = ['case_type_encoded', 'maternal_condition_encoded']
X = data_frame[features]
y = data_frame['underlying_cause_encoded']

# Ensure labels are correctly encoded as continuous integers starting from 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier()}

# Train and evaluate models
results = {}
feature_importances = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get predicted probabilities for ROC AUC calculation
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        decision_function = model.decision_function(X_test)
        y_pred_prob = np.exp(decision_function) / np.sum(np.exp(decision_function), axis=1, keepdims=True)
    else:
        y_pred_prob = None

    if y_pred_prob is not None and len(y_pred_prob.shape) == 2:
        auc_score = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    else:
        auc_score = None

    results[model_name] = {'accuracy': accuracy, 'auc': auc_score}
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances[model_name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[model_name] = model.coef_[0]

# Print results
for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['accuracy']}, AUC: {metrics['auc']}")

#4C Plot feature importance in descending order for each of the models using horizontal bar chart
for model_name, importances in feature_importances.items():
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=[features[i] for i in sorted_idx])
    plt.title(f"Feature Importance for {model_name}")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

# #5 Model Evaluation
# #5A Import the appropriate evaluation metric packages
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier

#5B Model Evaluation using cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{model_name} - Cross-Validated Accuracy: {cv_scores.mean()}")

# #5C Ensemble the models and see the performance of the combination models on the data
ensemble_model = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')
ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred)
ensemble_y_pred_prob = ensemble_model.predict_proba(X_test)
ensemble_auc = roc_auc_score(y_test, ensemble_y_pred_prob, multi_class='ovr')
print(f"Ensemble Model - Accuracy: {ensemble_accuracy}, AUC: {ensemble_auc}")

#5D Use Accuracy score metrics to evaluate the performance of the models
#Already included in 5B and 5C.

#5E Plot the AUC and ROC curve on the same graph to visualize and compare the performance of each of the models
# ROC and AUC plots
plt.figure(figsize=(12, 8))
for model_name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
        for i in range(y_pred_prob.shape[1]):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f'{model_name} class {i} (AUC = {auc(fpr, tpr):.2f})')
    elif hasattr(model, 'decision_function'):
        decision_function = model.decision_function(X_test)
        y_pred_prob = np.exp(decision_function) / np.sum(np.exp(decision_function), axis=1, keepdims=True)
        for i in range(y_pred_prob.shape[1]):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f'{model_name} class {i} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot AUC and ROC for ensemble model
plt.figure(figsize=(12, 8))
for i in range(ensemble_y_pred_prob.shape[1]):
    fpr, tpr, _ = roc_curve(y_test == i, ensemble_y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f'Ensemble Model class {i} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# #6 Visualization
# #6A Plot feature importance in descending order for each of the models using horizontal bar chart

for model_name, importances in feature_importances.items():
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=[features[i] for i in sorted_idx])
    plt.title(f"Feature Importance for {model_name}")
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

#6B Top five infant underlying causes of child death
top_infant_causes = data_frame['underlying_cause'].value_counts().head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_infant_causes.values, y=top_infant_causes.index)
plt.title('Top Five Infant Underlying Causes of Child Death')
plt.xlabel('Count')
plt.ylabel('Underlying Cause')
plt.show()

#6C Top five maternal factors contributing to child death
top_maternal_factors = data_frame['Main_maternal_condition'].value_counts().head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_maternal_factors.values, y=top_maternal_factors.index)
plt.title('Top Five Maternal Factors Contributing to Child Death')
plt.xlabel('Count')
plt.ylabel('Maternal Condition')
plt.show()

#6D Child death based on case types
case_type_counts = data_frame['case_type'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=case_type_counts.values, y=case_type_counts.index)
plt.title('Child Death Based on Case Types')
plt.xlabel('Count')
plt.ylabel('Case Type')
plt.show()

