**DATASET**

Dataset is downloaded from kaggle.
Link - https://www.kaggle.com/datasets/manishkc06/engineering-graduate-salary-prediction

It has 2998 rows × 34 columns.
It contains the below columns 
'ID', 'Gender', 'DOB', '10percentage', '10board', '12graduation','12percentage', '12board', 'CollegeID', 'CollegeTier', 'Degree',
'Specialization', 'collegeGPA', 'CollegeCityID', 'CollegeCityTier','CollegeState', 'GraduationYear', 'English', 'Logical', 'Quant',
'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon','ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg',
'CivilEngg', 'conscientiousness', 'agreeableness', 'extraversion','nueroticism', 'openess_to_experience', 'Salary'.

**CODE DETAILS:**

_Import Libraries:_ Utilize essential Python libraries such as pandas, numpy, seaborn, matplotlib.pyplot, and machine learning tools from sklearn.

_Load Dataset:_ Read the dataset (e.g., 'Engineering_graduate_salary.csv') into a DataFrame using pandas.

_Data Preprocessing:_
Remove unnecessary columns (e.g., 'ID', 'DOB', 'CollegeID').
Handle missing values appropriately.
Standardize categorical variables (e.g., unify similar categories in 'Specialization').


_Feature Engineering:_
Convert categorical variables into dummy/indicator variables using pd.get_dummies().
Scale numerical features for uniformity using StandardScaler from sklearn.preprocessing.
Split Data: Divide the dataset into training and testing sets with train_test_split from sklearn.model_selection.

_Model Training and Evaluation:_
Train models like SGDRegressor, DecisionTreeRegressor, and RandomForestRegressor from sklearn.
Evaluate model performance using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).
