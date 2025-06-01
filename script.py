#docker build -t autosklearn-image .
#docker run -it autosklearn-image
#docker run -it -v $(pwd):/app autosklearn-image
import os
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from pandas.util import hash_pandas_object

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from autosklearn.classification import AutoSklearnClassifier
from imblearn.under_sampling import NearMiss
from scipy.stats import wilcoxon

if __name__ == "__main__":
    df_reduced = pd.read_csv("df_sampled.csv")

    #================Escolha do modelo a ser usado================
    # Hiperparâmetros dos modelos a serem testados 
    param_grid_KNN = {
        'n_neighbors': [3, 5, 10],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'kd_tree']
    }

    param_grid_DT = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5]
    }

    param_grid_RF = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 30, 50],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5]
    }

    param_grid_XGB = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    # Arrays de métricas
    test_scores_KNN, precision_scores_KNN, recall_scores_KNN, f1_scores_KNN = [], [], [], []
    test_scores_DT, precision_scores_DT, recall_scores_DT, f1_scores_DT = [], [], [], []
    test_scores_RF_full, precision_scores_RF_full, recall_scores_RF_full, f1_scores_RF_full = [], [], [], []
    test_scores_XGB, precision_scores_XGB, recall_scores_XGB, f1_scores_XGB = [], [], [], []
    test_scores_AUTO, precision_scores_AUTO, recall_scores_AUTO, f1_scores_AUTO = [], [], [], []

    # KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scaler = StandardScaler()

    x = df_reduced.drop('label', axis=1)
    y = df_reduced['label']

    x_clean = x.replace([np.inf, -np.inf], np.nan)
    x_clean = x_clean.fillna(x_clean.mean())

    x1 = x_clean.values
    y1 = y.values

    model_metrics = {
        "KNN": {
            "accuracy": test_scores_KNN,
            "precision": precision_scores_KNN,
            "recall": recall_scores_KNN,
            "f1": f1_scores_KNN
        },
        "DT": {
            "accuracy": test_scores_DT,
            "precision": precision_scores_DT,
            "recall": recall_scores_DT,
            "f1": f1_scores_DT
        },
        "RF": {
            "accuracy": test_scores_RF_full,
            "precision": precision_scores_RF_full,
            "recall": recall_scores_RF_full,
            "f1": f1_scores_RF_full
        },
        "XGB": {
            "accuracy": test_scores_XGB,
            "precision": precision_scores_XGB,
            "recall": recall_scores_XGB,
            "f1": f1_scores_XGB
        },
        "AutoSklearn": {
            "accuracy": test_scores_AUTO,
            "precision": precision_scores_AUTO,
            "recall": recall_scores_AUTO,
            "f1": f1_scores_AUTO
        }
    }
    
#========================Trinando e avaliando os modelos========================

    for fold_index, (train_index, test_index) in enumerate(kf.split(x1)):
        X_train, X_test = x1[train_index], x1[test_index]
        Y_train, Y_test = y1[train_index], y1[test_index]
        
        #isso aqui é o nearmiss e salvando os dados descartados concatenando no conjunto de testes
        counts = Counter(Y_train)
        valid_classes = {cls: 5000 for cls in range(5) if counts.get(cls, 0) >= 5000}

        nm = NearMiss(sampling_strategy=valid_classes, version=2, n_neighbors=3, n_jobs=-1)
        X_resampled, Y_resampled = nm.fit_resample(X_train, Y_train)

        hash_train = hash_pandas_object(pd.DataFrame(X_train)).values
        hash_resampled = hash_pandas_object(pd.DataFrame(X_resampled)).values
        selected_mask = np.isin(hash_train, hash_resampled)

        X_discarded = X_train[~selected_mask]
        Y_discarded = Y_train[~selected_mask]

        X_test = np.concatenate((X_test, X_discarded), axis=0)
        Y_test = np.concatenate((Y_test, Y_discarded), axis=0)

        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, Y_train, random_state=42, test_size=0.2)

        #normalizaçao dos dados
        X_train_val = scaler.fit_transform(X_train_val) 
        X_test_val = scaler.transform(X_test_val)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
#===========================================================================================================================    
        # Treinamento e validação dos modelos
        
        # KNN
        knn = KNeighborsClassifier(n_jobs=-1)
        grid_knn = GridSearchCV(knn, param_grid_KNN, cv=3, scoring='f1_weighted', n_jobs=1)
        grid_knn.fit(X_train_val, y_train_val)
        knn_best = grid_knn.best_estimator_
        knn_best.fit(X_train, Y_train)
        y_pred = knn_best.predict(X_test)
        test_scores_KNN.append(accuracy_score(Y_test, y_pred))
        precision_scores_KNN.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_KNN.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_KNN.append(f1_score(Y_test, y_pred, average='weighted'))

        # Decision Tree
        dt = DecisionTreeClassifier(max_features='log2', random_state=42)
        grid_dt = GridSearchCV(dt, param_grid_DT, cv=3, scoring='f1_weighted', n_jobs=1)
        grid_dt.fit(X_train_val, y_train_val)
        dt_best = grid_dt.best_estimator_
        dt_best.fit(X_train, Y_train)
        y_pred = dt_best.predict(X_test)
        test_scores_DT.append(accuracy_score(Y_test, y_pred))
        precision_scores_DT.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_DT.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_DT.append(f1_score(Y_test, y_pred, average='weighted'))

        # Random Forest
        rf = RandomForestClassifier(max_features='sqrt', random_state=42, n_jobs=-1)
        grid_rf = GridSearchCV(rf, param_grid_RF, cv=3, scoring='f1_weighted', n_jobs=1)
        grid_rf.fit(X_train_val, y_train_val)
        rf_best = grid_rf.best_estimator_
        rf_best.fit(X_train, Y_train)
        y_pred = rf_best.predict(X_test)
        test_scores_RF_full.append(accuracy_score(Y_test, y_pred))
        precision_scores_RF_full.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_RF_full.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_RF_full.append(f1_score(Y_test, y_pred, average='weighted'))

        # XGBoost
        xgb = XGBClassifier(n_jobs=-1, random_state=42, eval_metric='logloss', tree_method='hist')
        grid_xgb = GridSearchCV(xgb, param_grid_XGB, cv=3, scoring='f1_weighted', n_jobs=1)
        grid_xgb.fit(X_train_val, y_train_val)
        xgb_best = grid_xgb.best_estimator_
        xgb_best.fit(X_train, Y_train)
        y_pred = xgb_best.predict(X_test)
        test_scores_XGB.append(accuracy_score(Y_test, y_pred))
        precision_scores_XGB.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_XGB.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_XGB.append(f1_score(Y_test, y_pred, average='weighted'))

        # AutoSklearn
        auto = AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60,ensemble_kwargs={"ensemble_size": 50},seed=42,n_jobs=-1)
        auto.fit(X_train, Y_train)
        y_pred = auto.predict(X_test)
        test_scores_AUTO.append(accuracy_score(Y_test, y_pred))
        precision_scores_AUTO.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_AUTO.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_AUTO.append(f1_score(Y_test, y_pred, average='weighted'))

    #métricas
    data = {
        "Modelo": ["KNN", "DT", "RF", "XGB", "AutoSKLearn"],
        "Acurácia": [np.mean(test_scores_KNN), np.mean(test_scores_DT), np.mean(test_scores_RF_full), np.mean(test_scores_XGB), np.mean(test_scores_AUTO)],
        "Precisão": [np.mean(precision_scores_KNN), np.mean(precision_scores_DT), np.mean(precision_scores_RF_full), np.mean(precision_scores_XGB), np.mean(precision_scores_AUTO)],
        "Recall": [np.mean(recall_scores_KNN), np.mean(recall_scores_DT), np.mean(recall_scores_RF_full), np.mean(recall_scores_XGB), np.mean(recall_scores_AUTO)],
        "F1 Score": [np.mean(f1_scores_KNN), np.mean(f1_scores_DT), np.mean(f1_scores_RF_full), np.mean(f1_scores_XGB), np.mean(f1_scores_AUTO)]
    }

    metrics = pd.DataFrame(data)
    metrics.to_csv("metrics.csv", index=False)

    # Wilcoxon
    model_names = list(model_metrics.keys())
    metrics_a = ["accuracy", "precision", "recall", "f1"]
    wilcoxon_results = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            for metric in metrics_a:
                data1 = model_metrics[model1][metric]
                data2 = model_metrics[model2][metric]

                stat, p_value = wilcoxon(data1, data2)
                wilcoxon_results[f"{model1} vs {model2} - {metric}"] = (stat, p_value)

    # Tabela de resultados
    table_data = [
        [comparison, stat, round(p, 3), "Sim" if p < 0.05 else "Não"]
        for comparison, (stat, p) in wilcoxon_results.items()
    ]

    df_results = pd.DataFrame(table_data, columns=["Comparação", "Estatística", "p-valor", "Rejeita H₀"])
    df_results.to_csv("wilcoxon_results.csv", index=False)
    
    print("\n ")
    print("\n ")
    print("Processamento concluído. Resultados salvos em 'metrics.csv' e 'wilcoxon_results.csv'.")
    print("Modelos treinados e salvos com sucesso.")
    # print("Dataset reduzido salvo como 'df_reduced.csv'.")
    # print(f"Nº de features selecionadas: {len(selected_features)}")
    print("Todos os processos concluídos com sucesso.")