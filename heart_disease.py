import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report , confusion_matrix , ConfusionMatrixDisplay , roc_auc_score , roc_curve
import numpy as np
from xgboost import XGBClassifier, plot_tree

def load_data():
    df = pd.read_csv("heart.csv")
    x = df.iloc[:, :-1]
    y = df['target']
    return(x, y, df)

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y ,test_size = 0.2, random_state = 42, stratify=y
    )
    return (x_train, x_test, y_train, y_test)

def class_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict = True)
    report = pd.DataFrame(report)
    plt.figure(figsize=(14, 6))
    sns.heatmap(report.iloc[: -1 , :-2], annot= True, fmt=".2f", cmap="Blues")
    plt.title("Heart Disease Classification Report")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.show()

def confu_matrix(y_test, y_pred):
    conf_forest = confusion_matrix(y_test, y_pred)
    disp_forest = ConfusionMatrixDisplay(confusion_matrix=conf_forest)
    disp_forest.plot(cmap="Blues")
    plt.title("Heart Disease Confusion Matrix")
    plt.tight_layout()
    plt.show()

def roc(model_xgb, x_test, y_test):
    plt.figure(figsize=(6, 4))
    fpr ,tpr, _ = roc_curve(y_test, model_xgb.predict_proba(x_test)[:, 1])
    plt.plot(fpr , tpr, label =f"Heart Disease AUC = {roc_auc_score(y_test, model_xgb.predict_proba(x_test)[:, 1]):.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Heart Disease Roc Curve")
    plt.legend()
    plt.show()

def show_xgboost(model_xgb):
    plot_tree(model_xgb ,
             num_trees=0, rankdir='LR'
             )
    plt.title("Heart Disease (Single Tree) Visualization", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

def set_model_xgboost(x_train, y_train, x_test):
    param = {
        'n_estimators':[50, 100, 200],
        'max_depth':[10, 15],
        'learning_rate':[0.01, 0.1, 0.2],
        'subsample':[0.8, 1.0],
        'colsample_bytree':[0.8, 1.0]
    }
    model_xgb = GridSearchCV(
        estimator = XGBClassifier(random_state = 42),
        param_grid = param,
        cv = 5, scoring = 'accuracy',
        n_jobs = -1
        )
    model_xgb.fit(x_train, y_train)
    model_xgb = model_xgb.best_estimator_
    y_pred = model_xgb.predict(x_test)
    y_pred = pd.DataFrame(data=y_pred)
    return (model_xgb, y_pred)

def show_importances(x, model_xgb):
    importances = model_xgb.feature_importances_
    feature_names = x.columns
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x = importances[indices], y= feature_names[indices])
    plt.title("Feature Importances for Heart Disease")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

def draw_categ(df):
    Categorical_feature = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    plt.figure(figsize=(15, 15))
    for i, column in enumerate(Categorical_feature, 1):
        plt.subplot(3, 3, i)
        df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
        df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
        plt.legend()
        plt.xlabel(column)
    plt.show()

def draw_conti(df):
    Conti_val = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    plt.figure(figsize=(15, 15))
    for i, column in enumerate(Conti_val, 1):
        plt.subplot(3, 2, i)
        df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
        df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
        plt.legend()
        plt.xlabel(column)
    plt.show()

def details(df):
    print(df.head())
    print(df.isna().sum())
    categorical_val = []
    continous_val = []
    for column in df.columns:
        print('-----------------------------')
        print(f"{column} : {df[column].unique()}")
        if len(df[column].unique()) <= 5:
            categorical_val.append(column)
        else:
            continous_val.append(column)
    print('-----------------------------')
    print(f"Categorical Features : {categorical_val}")
    print(f"Continous Features : {continous_val}")

def main():
    x, y, df= load_data()
    details(df)
    draw_categ(df)
    draw_conti(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model_xgb, y_pred = set_model_xgboost(x_train, y_train, x_test)
    class_report(y_test, y_pred)
    confu_matrix(y_test, y_pred)
    roc(model_xgb, x_test, y_test)
    show_xgboost(model_xgb)
    show_importances(x, model_xgb)

if __name__ == "__main__":
    main()