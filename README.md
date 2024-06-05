# Child Mortality Analysis

analysis of child mortality data from the CHAMPS dataset.

## Dataset

The dataset used in this project is the `CHAMPS.csv` file, which contains information about child mortality, including underlying causes and maternal conditions.

## Files

- `analysis.py`: This script performs the following tasks:
  - Loads and preprocesses the dataset
  - Performs exploratory data analysis (EDA)
  - Computes descriptive statistics
  - Conducts correlation analysis
  - Trains a logistic regression model
  - Visualizes the results

##  Dependencies
        - pandas
        - seaborn
        - scikit-learn
        - matplotlib
        - xgboost
    
## Results

### Exploratory Data Analysis

![Correlation Heatmap](src/output_charts/Correlation_Heatmap.png)

### Feature Importance

- Logistic Regression:
  ![Feature Importance for Logistic Regression](src/output_charts/Fig_4_1_Feature_Importance_for_Logistic_Regression.png)
  
- AdaBoost:
  ![Feature Importance for AdaBoost](src/output_charts/Fig_4_2_Feature_Importance_for_AdaBoost.png)
  
- Random Forest:
  ![Feature Importance for Random Forest](src/output_charts/Fig_4_3_Feature_Importance_for_Random_Forest.png)
  
- Gradient Boosting:
  ![Feature Importance for Gradient Boosting](src/output_charts/Fig_4_4%20_Feature_Importance_for_Gradient_Boosting.png)

- XGBoost:
  ![Feature Importance for XGBoost](src/output_charts/Fig_4_5_Feature_Importance_for_XGBoost.png)
  
### Top Five Infant Underlying Causes of Child Death

![Top Five Infant Underlying Causes of Child Death](src/output_charts/Fig_6_1_Top_Five_Infant_Underlying_Causes_of_Child_Death.png)

### Top Five Maternal Factors Contributing to Child Death

![Top Five Maternal Factors Contributing to Child Death](src/output_charts/Fig_6_2_Top_Five_Maternal_Factors_Contributing_to_Child_Death.png)

### Child Death Based on Case Types

![Child Death Based on Case Types](src/output_charts/Fig_6_3_Child_Death_Based_on_Case_Types.png)

### Model Evaluation

- AUC:
  ![AUC](src/output_charts/Fig_5_1_AUC.png)

- ROC:
  ![ROC](src/output_charts/Fig_5_2_ROC.png)


## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zbluee/EPHI_data_challenge
   cd  EPHI_data_challenge 
2. **install the Dependencies**:
```pip install pandas seaborn scikit-learn matplotlib xgboost```