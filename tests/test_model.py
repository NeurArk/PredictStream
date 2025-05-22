import pandas as pd
from sklearn.datasets import make_classification, make_regression

from utils import model


def sample_df():
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df


def sample_reg_df():
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df


def test_train_test_split():
    df = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(
        df, "target", test_size=0.2, random_state=42
    )
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_logistic_regression_training():
    df = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    clf = model.train_logistic_regression(X_train, y_train, max_iter=100)
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)


def test_random_forest_training():
    df = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    clf = model.train_random_forest_classifier(X_train, y_train, n_estimators=10)
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)


def test_cross_validation():
    df = sample_df()
    X = df.drop(columns=["target"])
    y = df["target"]
    clf = model.train_logistic_regression(X, y, max_iter=100)
    scores = model.cross_validate_model(clf, X, y, cv=3)
    assert len(scores) == 3


def test_model_caching():
    df = sample_df()
    X_train, _, y_train, _ = model.train_test_split_data(df, "target")

    @model.cache_model
    def custom_train(x, y):
        return model.train_logistic_regression(x, y, max_iter=50)

    first = custom_train(X_train, y_train)
    second = custom_train(X_train, y_train)
    assert first is second


def test_training_function_caching():
    df = sample_reg_df()
    X = df.drop(columns=["target"])
    y = df["target"]
    first = model.train_linear_regression(X, y)
    second = model.train_linear_regression(X, y)
    assert first is second


def test_regression_training_functions():
    df = sample_reg_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df, "target")
    lr = model.train_linear_regression(X_train, y_train)
    dt = model.train_decision_tree_regressor(X_train, y_train, max_depth=2)
    rf = model.train_random_forest_regressor(X_train, y_train, n_estimators=10)
    for reg in (lr, dt, rf):
        preds = reg.predict(X_test)
        assert len(preds) == len(y_test)


def test_problem_type_detector():
    df_class = sample_df()
    df_reg = sample_reg_df()
    assert model.detect_problem_type(df_class["target"]) == "classification"
    assert model.detect_problem_type(df_reg["target"]) == "regression"


def test_compare_models_and_save(tmp_path):
    df = sample_reg_df()
    X = df.drop(columns=["target"])
    y = df["target"]
    models_dict = {
        "lin": model.train_linear_regression(X, y),
        "tree": model.train_decision_tree_regressor(X, y, max_depth=2),
    }
    results = model.compare_models(models_dict, X, y, cv=3, scoring="r2")
    assert set(results) == {"lin", "tree"}
    path = tmp_path / "model.joblib"
    model.save_model(models_dict["lin"], path)
    assert path.exists() and path.stat().st_size > 0


def test_xgboost_training_functions():
    df_c = sample_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df_c, "target")
    xgb_clf = model.train_xgboost_classifier(
        X_train,
        y_train,
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
    )
    preds = xgb_clf.predict(X_test)
    assert len(preds) == len(y_test)

    df_r = sample_reg_df()
    X_train, X_test, y_train, y_test = model.train_test_split_data(df_r, "target")
    xgb_reg = model.train_xgboost_regressor(
        X_train,
        y_train,
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
    )
    preds = xgb_reg.predict(X_test)
    assert len(preds) == len(y_test)
