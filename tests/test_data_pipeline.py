"""Tests for data pipeline functions."""

import numpy as np
import pandas as pd


class TestDataCleaning:
    """Tests for data cleaning functions."""

    def test_sample_data_has_required_columns(self, sample_weather_data):
        """Test that sample data has all required columns."""
        required_columns = [
            "date",
            "min_temp",
            "max_temp",
            "mean_temp",
            "sunshine",
            "global_radiation",
            "precipitation",
            "pressure",
            "cloud_cover",
            "snow_depth",
        ]
        for col in required_columns:
            assert col in sample_weather_data.columns

    def test_sample_data_has_no_nulls(self, sample_weather_data):
        """Test that sample data has no null values."""
        assert sample_weather_data.isnull().sum().sum() == 0

    def test_date_column_is_datetime(self, sample_weather_data):
        """Test that date column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_weather_data["date"])

    def test_temperature_range_valid(self, sample_weather_data):
        """Test that temperature values are within reasonable range."""
        assert sample_weather_data["min_temp"].min() >= -50
        assert sample_weather_data["max_temp"].max() <= 50
        assert (sample_weather_data["max_temp"] >= sample_weather_data["min_temp"]).all()

    def test_sunshine_hours_valid(self, sample_weather_data):
        """Test that sunshine hours are within valid range."""
        assert sample_weather_data["sunshine"].min() >= 0
        assert sample_weather_data["sunshine"].max() <= 24

    def test_cloud_cover_valid(self, sample_weather_data):
        """Test that cloud cover is within valid range (0-10 oktas)."""
        assert sample_weather_data["cloud_cover"].min() >= 0
        assert sample_weather_data["cloud_cover"].max() <= 10

    def test_precipitation_non_negative(self, sample_weather_data):
        """Test that precipitation is non-negative."""
        assert sample_weather_data["precipitation"].min() >= 0


class TestChronologicalSplit:
    """Tests for chronological data splitting."""

    def test_split_maintains_order(self, sample_weather_data):
        """Test that chronological split maintains temporal order."""
        df = sample_weather_data.sort_values("date").reset_index(drop=True)

        # Simulate chronological split
        n = len(df)
        train_end = int(n * 0.7)
        valid_end = int(n * 0.85)

        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        test = df.iloc[valid_end:]

        # Check no overlap
        assert train["date"].max() < valid["date"].min()
        assert valid["date"].max() < test["date"].min()

    def test_split_sizes(self, sample_weather_data):
        """Test that splits have reasonable sizes."""
        n = len(sample_weather_data)
        train_end = int(n * 0.7)
        valid_end = int(n * 0.85)

        train_size = train_end
        valid_size = valid_end - train_end
        test_size = n - valid_end

        assert train_size > 0
        assert valid_size > 0
        assert test_size > 0
        assert train_size + valid_size + test_size == n


class TestFeatureBuilding:
    """Tests for feature building functions."""

    def test_feature_arrays_shape(self, sample_features):
        """Test that feature arrays have correct shape."""
        X, y = sample_features
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

    def test_feature_arrays_no_nan(self, sample_features):
        """Test that feature arrays have no NaN values."""
        X, y = sample_features
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_feature_arrays_dtype(self, sample_features):
        """Test that feature arrays have numeric dtype."""
        X, y = sample_features
        assert np.issubdtype(X.dtype, np.number)
        assert np.issubdtype(y.dtype, np.number)
