import pandas as pd
import matplotlib.pyplot as plt
from utils import viz


def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [5, 4, 3, 2, 1],
        'cat': ['a', 'b', 'a', 'b', 'a'],
    })


def test_histogram_and_density():
    df = sample_df()
    fig = viz.histogram(df, 'num1', bins=2, title='Hist')
    assert fig.layout.title.text == 'Hist'
    fig = viz.histogram(df, 'num1', density=True)
    assert fig.data[0].histnorm == 'probability density'


def test_scatter_plot():
    df = sample_df()
    fig = viz.scatter_plot(df, 'num1', 'num2', color='cat', title='Scatter')
    assert fig.layout.title.text == 'Scatter'
    assert fig.data[0].marker.color is not None


def test_bar_and_pie_charts():
    df = sample_df()
    bar = viz.bar_chart(df, 'cat', 'num1')
    pie = viz.pie_chart(df, names='cat', values='num1')
    assert bar.data and pie.data


def test_box_and_violin():
    df = sample_df()
    box = viz.box_plot(df, x='cat', y='num1')
    violin = viz.violin_plot(df, x='cat', y='num1')
    assert box.data and violin.data


def test_heatmap():
    df = sample_df()
    fig = viz.heatmap(df, title='Heat')
    assert fig.layout.title.text == 'Heat'


def test_export_figure(tmp_path):
    df = sample_df()
    fig = viz.bar_chart(df, 'cat', 'num1')
    out = tmp_path / 'chart.html'
    viz.export_figure(fig, out)
    assert out.exists() and out.stat().st_size > 0


def test_pair_plot_and_image_export(tmp_path):
    df = sample_df()
    fig = viz.pair_plot(df, hue='cat')
    assert isinstance(fig, plt.Figure)
    png = tmp_path / 'pair.png'
    jpg = tmp_path / 'pair.jpg'
    viz.export_figure(fig, png)
    viz.export_figure(fig, jpg)
    assert png.exists() and png.stat().st_size > 0
    assert jpg.exists() and jpg.stat().st_size > 0
