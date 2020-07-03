import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponent,
    IterativeComponentWithSampleWeight,
)
from autosklearn.pipeline.constants import *
from autosklearn.pipeline.implementations.util import softmax
from autosklearn.util.common import check_none, check_for_bool


class GradientBoostingClassifierWOEarlyStopping(
    IterativeComponent,
    AutoSklearnClassificationAlgorithm
):
    def __init__(self, loss, learning_rate, min_samples_leaf, max_depth,
                 max_leaf_nodes, max_bins, l2_regularization, scoring,
                 random_state=None,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = self.get_max_iter()
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    @staticmethod
    def get_max_iter():
        return 512

    def iterative_fit(self, X, y, n_iter=2, refit=False):

        """
        Set n_iter=2 for the same reason as for SGD
        """
        import sklearn.ensemble
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.learning_rate = float(self.learning_rate)
            self.max_iter = int(self.max_iter)
            self.min_samples_leaf = int(self.min_samples_leaf)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.max_bins = int(self.max_bins)
            self.l2_regularization = float(self.l2_regularization)
            self.verbose = int(self.verbose)
            n_iter = int(np.ceil(n_iter))

            # initial fit of only increment trees
            self.estimator = sklearn.ensemble.HistGradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                max_iter=n_iter,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                max_bins=self.max_bins,
                l2_regularization=self.l2_regularization,
                tol=None,
                scoring=self.scoring,
                n_iter_no_change=0,
                validation_fraction=None,
                verbose=self.verbose,
                warm_start=True,
                random_state=self.random_state,
            )
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter,
                                          self.max_iter)

        self.estimator.fit(X, y)

        if self.estimator.max_iter > self.estimator.n_iter_:
            raise ValueError()
        if self.estimator.max_iter >= self.max_iter:
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True)
        max_depth = UnParametrizedHyperparameter(
            name="max_depth", value="None")
        max_leaf_nodes = UniformIntegerHyperparameter(
            name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True)
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization", lower=1E-10, upper=1, default_value=1E-10, log=True)
        scoring = UnParametrizedHyperparameter(
            name="scoring", value="loss")

        cs.add_hyperparameters([loss, learning_rate, min_samples_leaf,
                                max_depth, max_leaf_nodes, max_bins, l2_regularization,
                                scoring,])

        return cs


class SGDWOEarlyStopping(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm,
):
    def __init__(self, loss, penalty, alpha, fit_intercept,
                 learning_rate, l1_ratio=0.15, epsilon=0.1,
                 eta0=0.01, power_t=0.5, average=False, random_state=None):
        self.max_iter = self.get_max_iter()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average

        self.estimator = None
        self.n_iter_ = None

    @staticmethod
    def get_max_iter():
        return 1024

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.linear_model import SGDClassifier

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None
            self.n_iter_ = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.alpha = float(self.alpha)
            self.l1_ratio = float(self.l1_ratio) if self.l1_ratio is not None \
                else 0.15
            self.epsilon = float(self.epsilon) if self.epsilon is not None \
                else 0.1
            self.eta0 = float(self.eta0)
            self.power_t = float(self.power_t) if self.power_t is not None \
                else 0.5
            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)

            self.estimator = SGDClassifier(loss=self.loss,
                                           penalty=self.penalty,
                                           alpha=self.alpha,
                                           fit_intercept=self.fit_intercept,
                                           max_iter=n_iter,
                                           tol=None,
                                           learning_rate=self.learning_rate,
                                           l1_ratio=self.l1_ratio,
                                           epsilon=self.epsilon,
                                           eta0=self.eta0,
                                           power_t=self.power_t,
                                           shuffle=True,
                                           average=self.average,
                                           random_state=self.random_state,
                                           warm_start=True)
            self.estimator.fit(X, y, sample_weight=sample_weight)
            self.n_iter_ = self.estimator.n_iter_
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, self.max_iter)
            self.estimator._validate_params()
            self.estimator._partial_fit(
                X, y,
                alpha=self.estimator.alpha,
                C=1.0,
                loss=self.estimator.loss,
                learning_rate=self.estimator.learning_rate,
                max_iter=n_iter,
                sample_weight=sample_weight,
                classes=None,
                coef_init=None,
                intercept_init=None
            )
            self.n_iter_ += self.estimator.n_iter_

        if self.estimator.max_iter > self.n_iter_:
            raise ValueError()
        if self.estimator.max_iter >= self.max_iter:
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.loss in ["log", "modified_huber"]:
            return self.estimator.predict_proba(X)
        else:
            df = self.estimator.decision_function(X)
            return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SGD Classifier',
                'name': 'Stochastic Gradient Descent Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        loss = CategoricalHyperparameter("loss",
            ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            default_value="log")
        penalty = CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default_value="l2")
        alpha = UniformFloatHyperparameter(
            "alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
        l1_ratio = UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1,  log=True, default_value=0.15)
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        epsilon = UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
        learning_rate = CategoricalHyperparameter(
            "learning_rate", ["optimal", "invscaling", "constant"],
            default_value="invscaling")
        eta0 = UniformFloatHyperparameter(
            "eta0", 1e-7, 1e-1, default_value=0.01, log=True)
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1,
                                             default_value=0.5)
        average = CategoricalHyperparameter(
            "average", ["False", "True"], default_value="False")
        cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept,
                                epsilon, learning_rate, eta0, power_t,
                                average])

        # TODO add passive/aggressive here, although not properly documented?
        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")

        power_t_condition = EqualsCondition(power_t, learning_rate,
                                            "invscaling")

        # eta0 is only relevant if learning_rate!='optimal' according to code
        # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
        # linear_model/sgd_fast.pyx#L603
        eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling",
                                                            "constant"])
        cs.add_conditions([elasticnet, epsilon_condition, power_t_condition,
                           eta0_in_inv_con])

        return cs


class PassiveAggressiveWOEarlyStopping(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm,
):
    def __init__(self, C, fit_intercept, loss, average, random_state=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.average = average
        self.loss = loss
        self.random_state = random_state
        self.estimator = None
        self.max_iter = self.get_max_iter()

    @staticmethod
    def get_max_iter():
        return 1024

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.linear_model import PassiveAggressiveClassifier

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.C = float(self.C)

            call_fit = True
            self.estimator = PassiveAggressiveClassifier(
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=n_iter,
                tol=None,
                loss=self.loss,
                shuffle=True,
                random_state=self.random_state,
                warm_start=True,
                average=self.average,
            )
            self.classes_ = np.unique(y.astype(int))
        else:
            call_fit = False

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator.max_iter = 50
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
            self.estimator.fit(X, y)
            self.fully_fit_ = True
        else:
            if call_fit:
                self.estimator.fit(X, y)
            else:
                self.estimator.max_iter += n_iter
                self.estimator.max_iter = min(self.estimator.max_iter, self.max_iter)
                self.estimator._validate_params()
                lr = "pa1" if self.estimator.loss == "hinge" else "pa2"
                self.estimator._partial_fit(
                    X, y,
                    alpha=1.0,
                    C=self.estimator.C,
                    loss="hinge",
                    learning_rate=lr,
                    max_iter=n_iter,
                    classes=None,
                    sample_weight=sample_weight,
                    coef_init=None,
                    intercept_init=None
                )
                if n_iter > self.estimator.n_iter_:
                    raise ValueError()
                if self.estimator.max_iter >= self.max_iter:
                    self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.decision_function(X)
        return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PassiveAggressive Classifier',
                'name': 'Passive Aggressive Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        loss = CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default_value="hinge"
        )

        # Note: Average could also be an Integer if > 1
        average = CategoricalHyperparameter('average', ['False', 'True'],
                                            default_value='False')

        cs = ConfigurationSpace()
        cs.add_hyperparameters([loss, fit_intercept, C, average])
        return cs
