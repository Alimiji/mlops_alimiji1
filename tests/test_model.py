"""Tests for model training and evaluation."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TestModelTraining:
    """Tests for model training."""

    def test_model_can_fit(self, sample_features):
        """Test that model can be trained on sample data."""
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "estimators_")
        assert len(model.estimators_) == 10

    def test_model_can_predict(self, sample_features):
        """Test that trained model can make predictions."""
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_model_predictions_reasonable(self, sample_features):
        """Test that predictions are within reasonable range."""
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        # Predictions should not be extremely different from training data
        assert predictions.min() > y.min() - 10
        assert predictions.max() < y.max() + 10


class TestModelEvaluation:
    """Tests for model evaluation metrics."""

    def test_rmse_calculation(self, sample_features):
        """Test RMSE calculation."""
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        assert rmse >= 0
        assert not np.isnan(rmse)

    def test_mae_calculation(self, sample_features):
        """Test MAE calculation."""
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)

        assert mae >= 0
        assert not np.isnan(mae)

    def test_r2_calculation(self, sample_features):
        """Test R2 calculation."""
        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        r2 = r2_score(y, predictions)

        # R2 should be between -inf and 1, but for a fitted model typically > 0
        assert r2 <= 1
        assert not np.isnan(r2)

    def test_train_r2_higher_than_test(self, sample_features):
        """Test that train R2 is typically higher than test R2 (overfitting check)."""
        X, y = sample_features

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))

        # Train R2 is usually higher (model overfits slightly)
        assert train_r2 >= test_r2 or abs(train_r2 - test_r2) < 0.2


class TestModelSerialization:
    """Tests for model serialization."""

    def test_model_can_be_saved_and_loaded(self, sample_features, tmp_path):
        """Test that model can be serialized and deserialized."""
        import joblib

        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)

        # Load
        loaded_model = joblib.load(model_path)

        # Compare predictions
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_loaded_model_has_same_params(self, sample_features, tmp_path):
        """Test that loaded model has same parameters."""
        import joblib

        X, y = sample_features
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)

        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)

        assert model.n_estimators == loaded_model.n_estimators
        assert model.max_depth == loaded_model.max_depth
        assert model.random_state == loaded_model.random_state
