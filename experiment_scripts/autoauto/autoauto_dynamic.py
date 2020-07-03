from collections import defaultdict
import json
import sys
from typing import Any, Dict, List, Optional

import ConfigSpace
import numpy as np
import pandas as pd
import sklearn.ensemble

sys.path.append('..')
import portfolio.portfolio_util


class OneVSOneSelector(object):
    def __init__(self, configuration, default_strategy_idx, rng):
        self.configuration = configuration
        self.default_strategy_idx = default_strategy_idx
        self.rng = rng
        self.models = None
        self.target_indices = None
        self.selectors_ = None
        self.weights_ = {}
        self.X_train = None

    def fit(self, X, y, methods, minima, maxima):
        self.X_train = X.copy()
        target_indices = np.array(list(range(y.shape[1])))
        models = dict()
        weights = dict()
        for i in range(len(target_indices)):
            models[i] = dict()
            weights[i] = dict()
            for j in range(i + 1, len(target_indices)):
                y_i_j = y[:, i] < y[:, j]
                min_i = np.array([minima[methods[i]][task_id] for task_id in X.index])
                max_i = np.array([maxima[methods[i]][task_id] for task_id in X.index])
                min_j = np.array([minima[methods[j]][task_id] for task_id in X.index])
                max_j = np.array([maxima[methods[j]][task_id] for task_id in X.index])

                minimum = np.minimum(min_i, min_j)
                maximum = np.maximum(max_i, max_j)
                diff = maximum - minimum
                diff[diff == 0] = 1
                normalized_y_i = (y[:, i].copy() - minimum) / diff
                normalized_y_j = (y[:, j].copy() - minimum) / diff

                weights_i_j = np.abs(normalized_y_i - normalized_y_j)
                if np.all([target == y_i_j[0] for target in y_i_j]):
                    n_zeros = int(np.ceil(len(y_i_j) / 2))
                    n_ones = int(np.floor(len(y_i_j) / 2))
                    base_model = sklearn.dummy.DummyClassifier(strategy='constant', constant=y_i_j[0])
                    base_model.fit(
                        X.values,
                        np.array(([[0]] * n_zeros) + ([[1]] * n_ones)).flatten(),
                        sample_weight=weights_i_j,
                    )
                else:
                    base_model = sklearn.ensemble.RandomForestClassifier(
                        random_state=self.rng,
                        n_estimators=500,
                        oob_score=True,
                        bootstrap=True,
                        min_samples_split=self.configuration['min_samples_split'],
                        min_samples_leaf=self.configuration['min_samples_leaf'],
                        max_features=int(np.rint(X.shape[1] ** self.configuration['max_features'])),
                    )
                    base_model.fit(X.values, y_i_j, sample_weight=weights_i_j)
                models[i][j] = base_model
                weights[i][j] = weights_i_j
        self.models = models
        self.weights_ = weights
        self.target_indices = target_indices

    def predict(self, X):

        if self.default_strategy_idx is not None:
            use_prediction = False
            counter = 0
            te = X.copy().flatten()
            assert len(te) == 3
            for _, tr in self.X_train.iterrows():
                tr = tr.to_numpy()
                if tr[0] >= te[0] and tr[1] >= te[1] and tr[2] >= te[2]:
                    counter += 1

            if counter > 0:
                use_prediction = True

            if not use_prediction:
                print('Backup', counter)
                return np.array([1 if i == self.default_strategy_idx else 0 for i in self.target_indices])
            print('No backup', counter)

        X = X.reshape((1, -1))

        raw_predictions = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                raw_predictions[(i, j)] = self.models[i][j].predict(X)

        predictions = []
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    prediction = raw_predictions[(i, j)][x_idx]
                    if prediction == 1:
                        wins[i] += 1
                    else:
                        wins[j] += 1
                    #prediction = raw_predictions[(i, j)][x_idx]
                    #wins[i] += prediction[1]
                    #wins[j] += prediction[0]
            wins = wins / np.sum(wins)
            predictions.append(wins)
        predictions = np.array([np.array(prediction) for prediction in predictions])
        return predictions

    def predict_oob(self, X):

        raw_predictions = dict()
        for i in range(len(self.target_indices)):
            for j in range(i + 1, len(self.target_indices)):
                rp = self.models[i][j].oob_decision_function_.copy()
                rp[np.isnan(rp)] = 0
                rp = np.nanargmax(rp, axis=1)
                raw_predictions[(i, j)] = rp

        predictions = []
        for x_idx in range(X.shape[0]):
            wins = np.zeros(self.target_indices.shape)
            for i in range(len(self.target_indices)):
                for j in range(i + 1, len(self.target_indices)):
                    prediction = raw_predictions[(i, j)][x_idx]
                    if prediction == 1:
                        wins[i] += 1
                    else:
                        wins[j] += 1
                    #prediction = raw_predictions[(i, j)][x_idx]
                    #wins[i] += prediction[1]
                    #wins[j] += prediction[0]
            wins = wins / np.sum(wins)
            predictions.append(wins)
        predictions = np.array([np.array(prediction) for prediction in predictions])
        return predictions


def build(
    performance_matrix: pd.DataFrame,
    matrices: Dict[str, Dict[str, np.ndarray]],
    metafeatures: pd.DataFrame,
    random_state: np.random.RandomState,
    task_ids: List[int],
    configurations: Dict[str, List[str]],
    seed: int,
):
    if performance_matrix.shape[1] != len(matrices):
        raise ValueError('mismatch in the data structure. Performance_matrix vs matrics contain '
                         'the following keys', performance_matrix.columns, matrices.keys())

    strategies = list(matrices.keys())

    minima_for_methods = dict()
    minima_for_tasks = dict()
    maxima_for_methods = dict()
    maxima_for_tasks = dict()

    for method in strategies:
        _, y_test, _, _, _ = portfolio.portfolio_util.reformat_data(
            matrices[method], task_ids, configurations[method])
        matrix = y_test.copy()
        minima = np.nanmin(np.nanmin(matrix, axis=2), axis=1)
        minima_as_dicts = {
            task_id: minima[i] for i, task_id in enumerate(metafeatures.index)
        }
        maxima = np.nanmax(np.nanmax(matrix, axis=2), axis=1)
        maxima_as_dicts = {
            task_id: maxima[i] for i, task_id in enumerate(metafeatures.index)
        }
        minima_for_methods[method] = minima_as_dicts
        maxima_for_methods[method] = maxima_as_dicts
        diff = maxima - minima
        diff[diff == 0] = 1
        del matrix

    for task_id in metafeatures.index:
        min_for_task = 1.0
        for method in strategies:
            min_for_task = min(min_for_task, minima_for_methods[method][task_id])
        minima_for_tasks[task_id] = min_for_task
        max_for_task = 0.0
        for method in strategies:
            max_for_task = max(max_for_task, maxima_for_methods[method][task_id])
        maxima_for_tasks[task_id] = max_for_task

    # Classification approach - generate data
    y_values = []
    task_id_to_idx = {}
    for i, task_id in enumerate(metafeatures.index):
        values = []
        task_id_to_idx[task_id] = len(y_values)
        for method in strategies:
            val = performance_matrix[method][task_id]
            values.append(val)
        y_values.append(values)
    y_values = np.array(y_values)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter('min_samples_split', 2, 20, log=True,
                                                 default_value=2)
    )
    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter('min_samples_leaf', 1, 20, log=True,
                                                 default_value=1)
    )
    cs.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter('max_features', 0, 1, default_value=0.5)
    )
    cs.seed(random_state.randint(0, 1000))
    configurations = [cs.get_default_configuration()] + cs.sample_configuration(size=50)

    default_strategies = [
        'RF_SH-eta4-i_holdout_iterative_es_if',
        "RF_None_holdout_iterative_es_if",
        "RF_SH-eta4-i_3CV_iterative_es_if",
        "RF_None_3CV_iterative_es_if",
        "RF_SH-eta4-i_5CV_iterative_es_if",
        "RF_None_5CV_iterative_es_if",
        "RF_SH-eta4-i_10CV_iterative_es_if",
        "RF_None_10CV_iterative_es_if"
    ]
    default_strategy = None
    for tmp in default_strategies:
        if tmp in strategies:
            default_strategy = tmp
            break
    if default_strategy is None:
        raise ValueError('Found no legal default strategy!')
    print('Using default strategy', default_strategy)

    best_loss = np.inf
    best_model = None
    best_sample_weight = None
    best_oob_predictions = None
    training_data = {}
    training_data['metafeatures'] = metafeatures.to_dict()
    training_data['y_values'] = [[float(_) for _ in row] for row in y_values]
    training_data['strategies'] = strategies
    training_data['minima_for_methods'] = minima_for_methods
    training_data['maxima_for_methods'] = maxima_for_methods

    for configuration in configurations:
        selector = OneVSOneSelector(
            configuration=configuration,
            default_strategy_idx=strategies.index(default_strategy),
            rng=random_state,
        )
        selector.fit(
            X=metafeatures,
            y=y_values,
            methods=strategies,
            minima=minima_for_methods,
            maxima=maxima_for_methods,
        )

        # # Training score
        # predictions = [selector.predict(row.to_numpy()) for _, row in metafeatures.iterrows()]
        # train_error = []
        # for i in range(len(predictions)):
        #     train_error_i = y_values[i][np.argmax(predictions[i])] != np.min(y_values[i])
        #     train_error.append(train_error_i)
        # train_error = np.array(train_error)
        #
        # sample_weight = []
        # for sample_idx, task_id in enumerate(metafeatures.index):
        #     prediction_idx = np.argmax(predictions[sample_idx])
        #     y_true_idx = np.argmin(y_values[sample_idx])
        #     diff = maxima_for_tasks[task_id] - minima_for_tasks[task_id]
        #     diff = 1 if diff == 0 else diff
        #     normalized_predicted_sample = (y_values[sample_idx, prediction_idx] - minima_for_tasks[
        #         task_id]) / diff
        #     normalized_y_true = (y_values[sample_idx, y_true_idx] - minima_for_tasks[
        #         task_id]) / diff
        #     weight = np.abs(normalized_predicted_sample - normalized_y_true)
        #     sample_weight.append(weight)
        # sample_weight = np.array(sample_weight)
        # train_loss = np.sum(train_error.astype(int) * sample_weight)

        # OOB score
        predictions = selector.predict_oob(metafeatures)
        error = []
        for i in range(len(predictions)):
            error_i = y_values[i][np.argmax(predictions[i])] != np.min(y_values[i])
            error.append(error_i)
        error = np.array(error)

        sample_weight = []
        for sample_idx, task_id in enumerate(metafeatures.index):
            prediction_idx = np.argmax(predictions[sample_idx])
            y_true_idx = np.argmin(y_values[sample_idx])
            diff = maxima_for_tasks[task_id] - minima_for_tasks[task_id]
            diff = 1 if diff == 0 else diff
            normalized_predicted_sample = (y_values[sample_idx, prediction_idx] - minima_for_tasks[task_id]) / diff
            normalized_y_true = (y_values[sample_idx, y_true_idx] - minima_for_tasks[task_id]) / diff
            weight = np.abs(normalized_predicted_sample - normalized_y_true)
            sample_weight.append(weight)
        sample_weight = np.array(sample_weight)
        loss = np.sum(error.astype(int) * sample_weight)

        # print(np.sum(train_error), np.sum(error), train_loss, loss, best_loss)
        if loss < best_loss:
            best_loss = loss
            best_model = selector
            best_sample_weight = sample_weight
            best_oob_predictions = predictions

    training_data['configuration'] = best_model.configuration.get_dictionary()
    with open('/tmp/training_data.json', 'wt') as fh:
        json.dump(training_data, fh, indent=4)

    # print('Best predictor OOB score', best_loss)

    # for i in best_model.models:
    #     for j in best_model.models[i]:
    #         print(best_model.models[i][j].feature_importances_)

    regrets_rf = []
    regret_random = []
    regret_oracle = []
    base_method_regets = {method: [] for method in strategies}
    # Normalize each column given the minimum and maximum ever observed on these tasks
    normalized_regret = np.array(y_values, dtype=float)
    for task_id in performance_matrix.index:
        task_idx = task_id_to_idx[task_id]
        minima = np.inf
        maxima = -np.inf
        for method in strategies:
            minima = min(minima_for_methods[method][task_id], minima)
            maxima = max(maxima_for_methods[method][task_id], maxima)
        diff = maxima - minima
        if diff == 0:
            diff = 1
        normalized_regret[task_idx] = (normalized_regret[task_idx] - minima) / diff

        prediction = best_oob_predictions[task_idx]
        prediction_idx = np.argmax(prediction)
        regrets_rf.append(float(normalized_regret[task_idx][prediction_idx]))
        regret_random.append(
            [float(value) for value in np.random.choice(normalized_regret[task_idx], size=1000, replace=True)]
        )
        regret_oracle.append(float(np.min(normalized_regret[task_idx])))
        for method_idx, method in enumerate(strategies):
            base_method_regets[method].append(normalized_regret[task_idx][method_idx])

    normalized_regret_dataframe = pd.DataFrame(normalized_regret,
                                               columns=performance_matrix.columns)
    full_oracle_perf = normalized_regret_dataframe.min(axis=1).mean()
    print('Oracle performance', full_oracle_perf)
    for i in range(normalized_regret_dataframe.shape[1]):
        subset_oracle_perf = normalized_regret_dataframe.drop(normalized_regret_dataframe.columns[i], axis=1).min(axis=1).mean()
        print(normalized_regret_dataframe.columns[i], subset_oracle_perf - full_oracle_perf)

    print('Regret rf', np.mean(regrets_rf))
    print('Regret random', np.mean(regret_random))
    print('Regret oracle', np.mean(regret_oracle))

    # all_regrets = {
    #     'selector': regrets_rf,
    #     'random': regret_random,
    #     'oracle': regret_oracle,
    #     'task_id_to_idx': task_id_to_idx,
    #     'base_methods': base_method_regets,
    # }
    # with open('/tmp/selector_training_data_regret_%d_only3feat.json' % seed, 'wt') as fh:
    #     json.dump(all_regrets, fh, indent=4)

    # for i, (_, mf) in enumerate(metafeatures.iterrows()):
    #     mf = mf.to_numpy().reshape((1, -1))
    #     prediction = best_model.predict(mf)
    #     oob_prediction = best_oob_predictions[i]
    #     print(oob_prediction, prediction, y_values[i], best_sample_weight[i],
    #           np.argmax(oob_prediction) == np.argmin(y_values[i]),
    #           np.argmax(prediction) == np.argmin(y_values[i]))

    return best_model
