from utils.data import sort_most_popular_platforms
import plotly.express as px
import itertools
import pandas as pd

def fig_calculate_channels(data_shop):
    colorscale = 'sunset' 
    count_all_platforms = {}

    for j in data_shop['tw_source_clean']:
        for source in j:
            if source in count_all_platforms:
                count_all_platforms[source] += 1
            else:
                count_all_platforms[source] = 1

    most_popular_platforms = sort_most_popular_platforms(count_all_platforms, 10)
    
    fig = px.bar(
        y=list(most_popular_platforms.keys()),
        x=list(most_popular_platforms.values()),
        orientation='h',
        title='Количество появлений платформ в путях клиентов',
        color=list(most_popular_platforms.keys()),  # Color based on x values
        color_continuous_scale=colorscale, labels={'x': 'Канал', 'y': 'Количество появлений'}

    )    
    return fig


def fig_purchase_prob(data_shop):
    colorscale = 'sunset'
    prob_buy = {}
    for i in range(1, 100):
        paths = data_shop[data_shop['len_tw_source'] == i].shape[0]
        if paths != 0:
            prob_buy[i] = data_shop[(data_shop['len_tw_source'] == i) & (data_shop['journey_success'] == 1)].shape[0] / paths
        else:
            prob_buy[i] = 0

    prob_buy = dict(itertools.islice(prob_buy.items(),20))
    
    fig = px.bar(
        x=list(prob_buy.keys()),
        y=list(prob_buy.values()),
        title='Вероятность покупки в зависимости от длины пути клиента',
        color=list(prob_buy.keys()),  # Color based on x values
        color_continuous_scale=colorscale,
        labels={'x': 'Длина пути', 'y': 'Вероятность покупки'}
    )
    return fig

def fig_attribution_result(attribution_result, attribution_name):
    """
    Args:
        attribution_result (dict): - dict with attribution result
        attribution_name (str): - name of the attribution
    Returns:
        fig (plotly.graph_objs._figure.Figure): - plotly figure
    """
    colorscale = 'plasma' #'sunset'  # Sunset color palette

    fig = px.bar(
        x=list(attribution_result.keys()),
        y=list(attribution_result.values()),
        title=f'Результат атрибуции {attribution_name}',
        color=list(attribution_result.keys()),  # Color based on x values
        color_continuous_scale=colorscale,
    )
    return fig
def draw_mmm_result(lr, X, y):
    """
    Builds an area chart showing the budget allocation across channels using MMM results.

    Args:
        lr (LinearRegression): Trained linear regression model.
        X (pd.DataFrame): Feature matrix used for training the model.
        y (pd.Series): Target variable used for training the model.

    Returns:
        plotly.graph_objs._figure.Figure: Area chart figure.
    """
    weights = pd.Series(lr.coef_, index=X.columns)
    base = lr.intercept_

    # Calculate unadjusted and adjusted contributions
    unadj_contributions = X.mul(weights).assign(Base=base)
    adj_contributions = (
        unadj_contributions
        .div(unadj_contributions.sum(axis=1), axis=0)
        .mul(y, axis=0)
    )

    fig = px.area(
        adj_contributions[['facebook-ads', 'google-ads', 'tiktok-ads', 
                            'pinterest-ads', 'snapchat-ads', 'amazon']],
        title='Распределение бюджета между каналами. MMM',
        labels={'value': 'Выручка', 'index': 'Дата'}
    )

    fig.update_layout(
        xaxis=dict(title='Дата'),
        yaxis=dict(title='Выручка'),
        legend_title='Каналы',
        legend=dict(
            title='Каналы',
            orientation='v',
            yanchor='middle',
            xanchor='right',
            x=1.01,
            y=0.5
        )
    )
    return fig

