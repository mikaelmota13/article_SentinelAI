#docker build -t autosklearn-image .
#docker run -it autosklearn-image
#docker run -it -v $(pwd):/app autosklearn-image

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from autosklearn.classification import AutoSklearnClassifier

from imblearn.under_sampling import NearMiss

from collections import Counter

if __name__ == "__main__":
    #df = pd.read_csv("df_sampled.csv")
    df_reduced = pd.read_csv("df_sampled.csv")
    # #fazendo a redução de colunas com MDI, fazendo o grid serache para isso.
    # x = df.drop('label', axis=1)
    # y = df['label']
    
    # x_clean = x.replace([np.inf, -np.inf], np.nan)
    # x_clean = x_clean.fillna(x_clean.mean())

    # x1 = x_clean.values
    # y1 = y.values

    # param_grid_RF = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    # kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # test_scores_RF_mdi, precision_scores_RF_mdi, recall_scores_RF_mdi, f1_scores_RF_mdi = [], [], [], []
    # feature_importances_list = []


    # for train_index, test_index in kf.split(x1):
    #     X_train, X_test = x1[train_index], x1[test_index]
    #     Y_train, Y_test = y1[train_index], y1[test_index]

    #     X_train_val, X_val, y_train_val, y_val = train_test_split(
    #         X_train, Y_train, random_state=42, test_size=0.2)

    #     best_acc = 0
    #     best_params = None

    #     for params in ParameterGrid(param_grid_RF):
    #         rf = RandomForestClassifier(
    #             n_estimators=params['n_estimators'],
    #             random_state=42,
    #             max_features='sqrt',
    #             n_jobs=-1
    #         )
    #         rf.fit(X_train_val, y_train_val)
    #         y_pred = rf.predict(X_val)

    #         acc = accuracy_score(y_val, y_pred)

    #         if acc > best_acc:
    #             best_acc = acc
    #             best_params = params

    #     rf_best = RandomForestClassifier(
    #         n_estimators=best_params['n_estimators'],
    #         random_state=42,
    #         max_features='sqrt',
    #         n_jobs=-1
    #     )
    #     rf_best.fit(X_train, Y_train)

    #     feature_importances_list.append(rf_best.feature_importances_)

    #     y_pred = rf_best.predict(X_test)

    #     test_scores_RF_mdi.append(accuracy_score(Y_test, y_pred))
    #     precision_scores_RF_mdi.append(precision_score(Y_test, y_pred, average='weighted'))
    #     recall_scores_RF_mdi.append(recall_score(Y_test, y_pred, average='weighted'))
    #     f1_scores_RF_mdi.append(f1_score(Y_test, y_pred, average='weighted'))


    # importances_mean = np.mean(feature_importances_list, axis=0)
    # importances_series = pd.Series(importances_mean, index=x.columns)

    # # Seleção de features pela mediana das importâncias
    # threshold = importances_series.median()
    # selected_features = importances_series[importances_series >= threshold].index.tolist()

    # # Dataset reduzido somente com as 
    # df_reduced = df[selected_features + ['label']]

    # metrics_mdi = pd.DataFrame({
    #     "Modelo": ["RandomForest"],
    #     "Acurácia": [np.mean(test_scores_RF_mdi)],
    #     "Precisão": [np.mean(precision_scores_RF_mdi)],
    #     "Recall": [np.mean(recall_scores_RF_mdi)],
    #     "F1 Score": [np.mean(f1_scores_RF_mdi)]
    # })

    # #metrics
    # #print(f"Features selecionadas: {selected_features}")
    # #print(f"Nº de features selecionadas: {len(selected_features)}")
    # metrics_mdi.to_csv("metrics_mdi.csv", index=False)
    # df_reduced.to_csv("df_reduced.csv", index=False)

    #================Escolha do modelo a ser usado================

    # Hiperparâmetros dos modelos a serem testados 
    param_grid_KNN = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'braycurtis', 'correlation']
    }
    param_grid_DT = {'max_depth': [3, 6, 7, 9, 11]}
    param_grid_RF = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    param_grid_XGB = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Arrays de métricas
    test_scores_KNN, precision_scores_KNN, recall_scores_KNN, f1_scores_KNN, accs_KNN, par_KNN = [], [], [], [], [], []
    test_scores_DT, precision_scores_DT, recall_scores_DT, f1_scores_DT, accs_DT, par_DT = [], [], [], [], [], []
    test_scores_RF_full, precision_scores_RF_full, recall_scores_RF_full, f1_scores_RF_full, accs_RF_full, par_RF_full = [], [], [], [], [], []
    test_scores_XGB, precision_scores_XGB, recall_scores_XGB, f1_scores_XGB, accs_XGB, par_XGB = [], [], [], [], [], []
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
        
        counts = Counter(Y_train)
        valid_classes = {cls: 50 for cls in range(5) if counts.get(cls, 0) >= 50}

        #if len(valid_classes) < 2:
            #continue  # pula fold com amostragem inválida

        nm = NearMiss(sampling_strategy=valid_classes, version=2, n_neighbors=3, n_jobs=-1)
        X_resampled, Y_resampled = nm.fit_resample(X_train, Y_train)

        #cria uma mascara booleana para selecionar as linhas de X_train que estão em X_resampled
        selected_mask = np.zeros(len(X_train), dtype=bool)
        for row in X_resampled:
            idx = np.where((X_train == row).all(axis=1))[0]
            if idx.size > 0:
                selected_mask[idx[0]] = True

        X_discarded = X_train[~selected_mask]
        Y_discarded = Y_train[~selected_mask]

        X_test = np.concatenate((X_test, X_discarded), axis=0)
        Y_test = np.concatenate((Y_test, Y_discarded), axis=0)
            
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, Y_train, random_state=42, test_size=0.2)

        # KNN
        for params in ParameterGrid(param_grid_KNN):
            X_train_val_KNN = scaler.fit_transform(X_train_val)
            X_test_val_KNN = scaler.transform(X_test_val)
            knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], metric=params['metric'], n_jobs=-1)
            knn.fit(X_train_val_KNN, y_train_val)
            y_pred = knn.predict(X_test_val_KNN)
            acc = accuracy_score(y_test_val, y_pred)
            accs_KNN.append(acc)
            par_KNN.append(params)

        X_train_KNN = scaler.fit_transform(X_train)
        X_test_KNN = scaler.transform(X_test)
        best_params_KNN = par_KNN[accs_KNN.index(max(accs_KNN))]
        knn_best = KNeighborsClassifier(metric=best_params_KNN['metric'], n_neighbors=best_params_KNN['n_neighbors'], n_jobs=-1)
        knn_best.fit(X_train_KNN, Y_train)
        y_pred = knn_best.predict(X_test_KNN)
        test_scores_KNN.append(accuracy_score(Y_test, y_pred))
        precision_scores_KNN.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_KNN.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_KNN.append(f1_score(Y_test, y_pred, average='weighted'))

        # Decision Tree
        for max_depth in param_grid_DT['max_depth']:
            X_train_val_DT = scaler.fit_transform(X_train_val)
            X_test_val_DT = scaler.transform(X_test_val)
            dt = DecisionTreeClassifier(max_depth=max_depth, max_features='log2', random_state=42)
            dt.fit(X_train_val_DT, y_train_val)
            y_pred = dt.predict(X_test_val_DT)
            acc = accuracy_score(y_test_val, y_pred)
            accs_DT.append(acc)
            par_DT.append(max_depth)

        X_train_DT = scaler.fit_transform(X_train)
        X_test_DT = scaler.transform(X_test)
        dt_best = DecisionTreeClassifier(max_depth=par_DT[accs_DT.index(max(accs_DT))], max_features='log2', random_state=42)
        dt_best.fit(X_train, Y_train)
        y_pred = dt_best.predict(X_test)
        test_scores_DT.append(accuracy_score(Y_test, y_pred))
        precision_scores_DT.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_DT.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_DT.append(f1_score(Y_test, y_pred, average='weighted'))

        # Random Forest
        for params in ParameterGrid(param_grid_RF):
            X_train_val_RF = scaler.fit_transform(X_train_val)
            X_test_val_RF = scaler.transform(X_test_val)
            rf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42, max_features='sqrt', n_jobs=-1, verbose=1)
            rf.fit(X_train_val_RF, y_train_val)
            y_pred = rf.predict(X_test_val_RF)
            acc = accuracy_score(y_test_val, y_pred)
            accs_RF_full.append(acc)
            par_RF_full.append(params)

        X_train_RF = scaler.fit_transform(X_train)
        X_test_RF = scaler.transform(X_test)
        best_param_RF = par_RF_full[accs_RF_full.index(max(accs_RF_full))]
        rf_best = RandomForestClassifier(**best_param_RF, random_state=42, max_features='sqrt', n_jobs=-1, verbose=1)
        rf_best.fit(X_train, Y_train)
        y_pred = rf_best.predict(X_test)
        test_scores_RF_full.append(accuracy_score(Y_test, y_pred))
        precision_scores_RF_full.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_RF_full.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_RF_full.append(f1_score(Y_test, y_pred, average='weighted'))

        # XGBoost
        for params in ParameterGrid(param_grid_XGB):
            X_train_val_XGB = scaler.fit_transform(X_train_val)
            X_test_val_XGB = scaler.transform(X_test_val)
            xgb = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], n_jobs=-1, random_state=42, eval_metric='logloss', tree_method='hist')
            xgb.fit(X_train_val_XGB, y_train_val)
            y_pred = xgb.predict(X_test_val_XGB)
            acc = accuracy_score(y_test_val, y_pred)
            accs_XGB.append(acc)
            par_XGB.append(params)

        X_train_XGB = scaler.fit_transform(X_train)
        X_test_XGB = scaler.transform(X_test)
        best_params = par_XGB[accs_XGB.index(max(accs_XGB))]
        xgb_best = XGBClassifier(**best_params, random_state=42, eval_metric='logloss', tree_method='hist')
        xgb_best.fit(X_train, Y_train)
        y_pred = xgb_best.predict(X_test)
        test_scores_XGB.append(accuracy_score(Y_test, y_pred))
        precision_scores_XGB.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_XGB.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_XGB.append(f1_score(Y_test, y_pred, average='weighted'))

        # AutoSklearn
        X_train_AUTO = scaler.fit_transform(X_train)
        X_test_AUTO = scaler.transform(X_test)
        auto = AutoSklearnClassifier(
            time_left_for_this_task=600,
            per_run_time_limit=60,
            ensemble_kwargs={"ensemble_size": 50},
            seed=42,
            n_jobs=-1
        )
        auto.fit(X_train_AUTO, Y_train)
        y_pred = auto.predict(X_test_AUTO)
        test_scores_AUTO.append(accuracy_score(Y_test, y_pred))
        precision_scores_AUTO.append(precision_score(Y_test, y_pred, average='weighted'))
        recall_scores_AUTO.append(recall_score(Y_test, y_pred, average='weighted'))
        f1_scores_AUTO.append(f1_score(Y_test, y_pred, average='weighted'))
        
        # os.makedirs("models", exist_ok=True)

        # joblib.dump(knn_best, f"model_knn_fold{fold_index}.joblib")
        # joblib.dump(dt_best, f"model_dt_fold{fold_index}.joblib")
        # joblib.dump(rf_best, f"model_rf_fold{fold_index}.joblib")
        # joblib.dump(xgb_best, f"model_xgb_fold{fold_index}.joblib")
        # joblib.dump(auto, f"model_autosklearn_fold{fold_index}.joblib")

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

    # Adiciona AutoSklearn nos resultados
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
        "AutoSKLearn": {
            "accuracy": test_scores_AUTO,
            "precision": precision_scores_AUTO,
            "recall": recall_scores_AUTO,
            "f1": f1_scores_AUTO
        }
    }

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