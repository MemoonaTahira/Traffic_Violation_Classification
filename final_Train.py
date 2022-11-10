import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import bentoml
import joblib
import wget
import json


wget.download("https://github.com/MemoonaTahira/Traffic_Violation_Classification/releases/download/latest/cleaned_traffic_violations.csv", out = "cleaned_traffic_violations.csv")


df = pd.read_csv("cleaned_traffic_violations.csv")
print('Dataframe dimensions:', df.shape)

# Target variable:
# Safety Equipment Repair Orders (SERO)
# Electric Vehicle safety equipment repair orders 
# To Combine SERO and ESERO into ERO becuase Sero is less the 0.01 of dataset, not even in imbalanced classification range
# whereas ESERO is about 5% of the dataset
df['violation_type'] = df['violation_type'].apply(lambda x:x if (x!="sero" and x!="esero") else "ero")
print(df.violation_type.value_counts(normalize=True))

# label encoding the target vairable:
target = df["violation_type"]
le = LabelEncoder()
target = le.fit_transform(target)

# Feature dataframe
features = df.drop(columns = "violation_type").copy()

# train-val-test split
x_train_full, x_test, y_train_full, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)
splits = [x_train_full, x_train, x_val, x_test]
for split in splits:
        split.reset_index(drop=True, inplace=True)

# creating model pipeline to impute missing values, scale numerical feature, reduce dimaensionality of categorical 
# variables (keeping a maximum of 30 categories in each feature) and and OHE encode the final categories, 
# and then selecting best 100 features based on mutual information classif score:
def create_model_pipeline(use_cat_OHE = False):

    categorical_columns = x_train_full.select_dtypes(include=["object_", "category"]).columns.tolist()
    boolean_columns = x_train_full.select_dtypes(include=["number"]).columns.tolist()
    boolean_columns.remove("car_age")
    numerical_columns = ["car_age"]
    boolean_transformer = SimpleImputer(strategy='most_frequent') 
    numerical_imputer = SimpleImputer(strategy="mean")
    numerical_robust_scaler = RobustScaler()
    numerical_transformer = Pipeline(
            steps = [("numerical_imputer", numerical_imputer),("scaler", numerical_robust_scaler),],
            verbose = True
    )
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_OHE_encoder = OneHotEncoder(max_categories = 30, drop = None, sparse=True, handle_unknown="ignore")

    # deciding whether to encode categroical feature as OHE or ordinal features
    if use_cat_OHE == True:
        categorical_transformer = Pipeline(
            steps = [("categorical_imputer", categorical_imputer),("OHE_encoder", categorical_OHE_encoder),
            ],
            verbose = True
        )

    else:
        categorical_transformer = categorical_imputer

    preprocess = ColumnTransformer(
        transformers=[("bool", boolean_transformer, boolean_columns),
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        verbose = True, remainder='passthrough', n_jobs = -1
    )
    
    # toggling feature selection on and off:

    model_pipeline = Pipeline(
        steps = [
            ('preprocess', preprocess),
            # ('vectorizer', DictVectorizer())
            ('feature_selection', SelectKBest(score_func = mutual_info_classif, k =100)),
            
        ],
        verbose = True
    )
    return model_pipeline


# create pipeline with fine-tuned hyper-paramter of the model:
model_pipeline = create_model_pipeline(use_cat_OHE = True)
tuned_model_classification = XGBClassifier(objective='multi:softmax',learning_rate=0.03, max_depth=20,n_estimators=200, eval_metric=balanced_accuracy_score, n_jobs = -1, random_state=42)
# model_pipeline.steps.insert(2,('classification', tuned_model_classification)) #insert as second step

# train with fine tuned hyper-params:
X_full_t = model_pipeline.fit_transform(x_train_full, y_train_full)
tuned_model_classification.fit(X_full_t, y_train_full)

filename = 'xgboost_traffic_violation_model.sav'
joblib.dump(tuned_model_classification, filename)
joblib.dump(model_pipeline, 'sklearn_pipeline.pkl')


loaded_model = joblib.load(filename)
print(loaded_model)

bentoml.xgboost.save_model(
    'traffic_violation_classification',
    tuned_model_classification,
    custom_objects={
        'model_pipeline': model_pipeline
    },
    signatures= {
        "predict":{
            "batchable": False,
            # we are going to concatenate arrays by first dimension
            # "batch_dim" : 0,
            
        }
    })



request = x_test.iloc[0].to_dict()
print("sample user data:", json.dumps(request, indent=2))

