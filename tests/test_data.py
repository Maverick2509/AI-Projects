import pandas as pd
import pytest

from src.data.prepare import clean_data


def test_target_is_binary():
    df = pd.DataFrame({
        "age": [45, 55, 65],
        "target": [0, 1, 2],
    })

    result = clean_data(df)

    assert set(result["target"].unique()) == {0, 1}


def test_duplicate_rows_are_removed():
    df = pd.DataFrame({
        "age": [45, 45],
        "target": [0, 0],
    })

    result = clean_data(df)

    assert len(result) == 1


def test_missing_feature_is_preserved_for_imputation():
    df = pd.DataFrame({
        "age": ["?", "55"],
        "target": [0, 1],
    })

    result = clean_data(df)

    assert result["age"].isna().sum() == 1


def test_target_column_is_required():
    df = pd.DataFrame({
        "age": [45, 55]
    })

    with pytest.raises(ValueError):
        clean_data(df)