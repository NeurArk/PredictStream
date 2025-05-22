import pandas as pd
import pytest

from utils import config
from utils import data


def test_load_data_csv(tmp_path):
    df_exp = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv_file = tmp_path / 'test.csv'
    df_exp.to_csv(csv_file, index=False)
    df = data.load_data(csv_file)
    pd.testing.assert_frame_equal(df, df_exp)


def test_load_data_excel(tmp_path):
    df_exp = pd.DataFrame({'a': [1, 2]})
    xls_file = tmp_path / 'test.xlsx'
    df_exp.to_excel(xls_file, index=False)
    df = data.load_data(xls_file)
    pd.testing.assert_frame_equal(df, df_exp)


def test_load_data_invalid(tmp_path):
    file = tmp_path / 'bad.txt'
    file.write_text('x')
    with pytest.raises(ValueError):
        data.load_data(file)


def test_convert_dtypes():
    df = pd.DataFrame({'num': ['1', '2'], 'date': ['2020-01-01', '2020-01-02']})
    conv = data.convert_dtypes(df)
    assert conv['num'].dtype.kind in {'i', 'f'}
    assert pd.api.types.is_datetime64_any_dtype(conv['date'])


def test_data_summary():
    df = pd.DataFrame({'a': [1, 2, 3]})
    summary = data.data_summary(df)
    assert 'a' in summary.columns


def test_sample_dataset_loads():
    for path in config.SAMPLE_DATASETS.values():
        df = data.load_data(path)
        assert not df.empty


def test_load_data_invalid_type():
    with pytest.raises(ValueError):
        data.load_data(123)


def test_convert_dtypes_invalid():
    with pytest.raises(TypeError):
        data.convert_dtypes([1, 2, 3])


def test_data_summary_empty():
    with pytest.raises(ValueError):
        data.data_summary(pd.DataFrame())


def test_validate_file_type(tmp_path):
    file = tmp_path / "a.csv"
    file.write_text("a,b\n1,2")
    ext = data.validate_file_type(file, ["csv", "xlsx"])
    assert ext == ".csv"


def test_validate_file_type_invalid(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("x")
    with pytest.raises(ValueError):
        data.validate_file_type(file, ["csv"])


def test_process_uploaded_file(tmp_path):
    import streamlit as st

    df = pd.DataFrame({"a": [1]})
    file = tmp_path / "test.csv"
    df.to_csv(file, index=False)
    st.session_state.clear()
    data.process_uploaded_file(file, session_key="up")
    pd.testing.assert_frame_equal(st.session_state["up"], df)


def test_upload_data_to_session(monkeypatch, tmp_path):
    import streamlit as st

    df = pd.DataFrame({"a": [1]})
    file = tmp_path / "test.csv"
    df.to_csv(file, index=False)
    st.session_state.clear()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: file)
    data.upload_data_to_session("Upload", session_key="foo")
    pd.testing.assert_frame_equal(st.session_state["foo"], df)

