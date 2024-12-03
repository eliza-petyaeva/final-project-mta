# Standard library imports
import ast
import itertools
import pickle

# Third-party imports
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Local module imports
from utils.etl import download_csv
from utils.model import train_mmm_model

from utils.attribution import (
    last_touch_attribution,
    first_touch_attribution,
    time_decay_attribution,
    position_based_attribution,
    markov_attribution,
    shapley_attribution,
    mean_channel_attribution_time,
    sorted_mean_channel_attribution_time,
)

from utils.data import (
    prep_data_for_markov_shapley,
    prepare_data,
    sort_attribution_result,
)

from utils.plots import (
    fig_calculate_channels,
    fig_purchase_prob,
    draw_mmm_result,
)

st.sidebar.title("MTA Веб-приложение")
st.write('''Это веб-приложение для анализа атрибуции рекламных кампаний. 
            Вы можете выбрать магазин и количество каналов для визуализации. 
            Также вы можете выбрать модели атрибуции для сравнения.
            В рамках MVP доступны модели: First Touch Attribution, Last Touch Attribution, Linear Attribution, Time Decay Attribution, Position Based Attribution,
            Markov Chain Attribution, Shapley Attribution, MMM Attribution. Вы можете посчитать атрибуцию в рамках предрождественской недели США.''')

# TODO: add download button for data

# Data loading section
st.sidebar.title("Загрузка данных")
file = st.sidebar.file_uploader("Загрузите файл в формате pickle", type="pickle")

if file is not None:
    data = pd.read_pickle(file)
    st.success("Данные успешно загружены.")
else:
    st.warning("Пожалуйста загрузите данные.")
    data = pd.read_pickle('data/data_for_mvp.pickle').reset_index(drop=True)

data_revenue = pd.read_csv('data/final_final_mmm.csv')
data['journey_end_ts'] = pd.to_datetime(data['journey_end_ts']).dt.date
data = data[data['len_tw_source'] <= 15]
data_shop_1, data_shop_2 = prepare_data(data)
models = ['fta', 'lta','linear', 'time-decay', 'position-based', 'markov', 'shapley']

# Interactive sidebar
selected_shop = st.sidebar.selectbox('Выберите магазин', ['beauty_shop_1', 'beauty_shop_2'])
selected_number = st.sidebar.slider('Количество платформ/каналов для визуализации', 5, 20) 

# Date range selection
st.sidebar.write('В данном разделе вы можете выбрать период для расчета атрибуции. Ремарка: данные доступны с 2022-12-17 по 2022-12-24.')
start_date = st.sidebar.date_input('Дата начала периода расчета атрибуции', pd.to_datetime('2022-12-17'))
end_date = st.sidebar.date_input('Дата конца периода расчета атрибуции', pd.to_datetime('2022-12-24'))
if start_date > end_date:
    st.error('Error: End Date must be after Start Date.')

# Display the selected dates
st.write('Дата начала периода:', start_date)
st.write('Дата конца периода:', end_date)

selected_models = st.multiselect('Выберите модели атрибуции:', models)



# Plot based on selected data shop
if selected_shop == 'beauty_shop_1':
    data_shop = data_shop_1.copy()
elif selected_shop == 'beauty_shop_2':
    data_shop = data_shop_2.copy()


fig_channels = fig_calculate_channels(data_shop)
fig_prob = fig_purchase_prob(data_shop)
data_shop = data_shop[(data_shop['journey_end_ts'] >= start_date) & (data_shop['journey_end_ts'] <= end_date)]
data_revenue = data_revenue[data_revenue['provider_account'] == selected_shop]
data_revenue['spendings'] = data_revenue['facebook-ads'] + data_revenue['google-ads'] + data_revenue['pinterest-ads'] + data_revenue['snapchat-ads'] + data_revenue['tiktok-ads'] + data_revenue['amazon']
all_channels, data_dic_timing = mean_channel_attribution_time(data_shop)
mcat = mean_channel_attribution_time(data_shop)
mean_time, mean_place, times = sorted_mean_channel_attribution_time(mcat)
# attribution_results = {}
all_channels = [data_shop['tw_source_clean'][i] for i in data_shop['tw_source_clean'].index]
all_channels = list(set(list(np.concatenate(all_channels).flat)))
data_to_download  = pd.DataFrame(index = all_channels)

# Plot based on selected date range
fig = go.Figure()
fig_revenue = go.Figure()
# first touch attribution
if 'fta' in selected_models:   
    fta= sort_attribution_result(first_touch_attribution(data_shop, selected_shop)).items()
    data_to_download['fta'] = dict(fta)
    fta_result = dict(itertools.islice(fta, selected_number))
    fig.add_trace(go.Bar(x=list(fta_result.keys()), 
                         y=list(fta_result.values()), 
                         name=f'First Touch Attribution'))
# last touch attribution
if 'lta' in selected_models:
    lta = sort_attribution_result(last_touch_attribution(data_shop, selected_shop)).items()
    data_to_download['lta'] = dict(lta)
    lta_result = dict(itertools.islice(lta, selected_number))
    fig.add_trace(go.Bar(x=list(lta_result.keys()), 
                         y=list(lta_result.values()), 
                         name=f'Last Touch Attribution'))
# linear attribution
if 'linear' in selected_models:
    linear = sort_attribution_result(first_touch_attribution(data_shop, selected_shop)).items()
    data_to_download['linear'] = dict(linear)
    linear_result = dict(itertools.islice(linear, selected_number))
    fig.add_trace(go.Bar(x=list(linear_result.keys()), 
                         y=list(linear_result.values()), 
                         name=f'Linear Attribution'))
# time decay attribution
if 'time-decay' in selected_models:
    time_decay = sort_attribution_result(time_decay_attribution(data_shop, selected_shop)).items()
    data_to_download['time-decay'] = dict(time_decay)
    time_decay_result = dict(itertools.islice(time_decay, selected_number))
    fig.add_trace(go.Bar(x=list(time_decay_result.keys()), 
                         y=list(time_decay_result.values()), 
                         name=f'Time Decay Attribution'))
# position based attribution
if 'position-based' in selected_models:
    position_based = sort_attribution_result(position_based_attribution(data_shop, selected_shop)).items()
    data_to_download['position-based'] = dict(position_based)
    position_based_result = dict(itertools.islice(position_based, selected_number))
    fig.add_trace(go.Bar(x=list(position_based_result.keys()), 
                         y=list(position_based_result.values()), 
                         name=f'Position Based Attribution'))
# markov attribution
if 'markov' in selected_models:
    mta_data_shop, budget = prep_data_for_markov_shapley(data_shop) # TODO: оптимизировать подсчет датасета
    markov = markov_attribution(mta_data_shop, budget)
    markov = sort_attribution_result(markov[0]['data']['conversion']).items()
    data_to_download['markov'] = dict(markov)
    markov = dict(itertools.islice(
        markov, 
        selected_number))
    fig.add_trace(go.Bar(x=list(markov.keys()), 
                         y=list(markov.values()), 
                         name=f'Markov Attribution'))
# shapley attribution
if 'shapley' in selected_models:
    mta_data_shop, budget = prep_data_for_markov_shapley(data_shop)
    shapley = shapley_attribution(mta_data_shop, budget)
    shapley = sort_attribution_result(shapley[0]['data']['conversion']).items()
    data_to_download['shapley'] = dict(shapley)
    shapley = dict(itertools.islice(
        shapley, 
        selected_number))
    fig.add_trace(go.Bar(x=list(shapley.keys()), 
                         y=list(shapley.values()), 
                         name=f'Shapley Attribution'))
    # st.write('Shapley Attribution:')

# Download the attribution results as a CSV file
download_link = download_csv(np.round(data_to_download.fillna(0), 3), f'attribution_results_{selected_shop}_{start_date}_{end_date}.csv')
st.markdown(download_link, unsafe_allow_html=True)


# Revenue and spendings plot
fig_revenue.add_trace(go.Scatter(mode='lines',x=data_revenue['event_date'], 
                                              y=data_revenue['total_price_usd'], 
                                              name=f'Выручка от заказов '))
fig_revenue.add_trace(go.Scatter(mode='lines',x=data_revenue['event_date'], 
                                              y=data_revenue['facebook-ads'], 
                                              name=f'Затраты на рекламу facebook-ads'))
fig_revenue.add_trace(go.Scatter(mode='lines',x=data_revenue['event_date'], 
                                              y=data_revenue['google-ads'], 
                                              name=f'Затраты на рекламу google-ads'))
fig_revenue.add_trace(go.Scatter(mode='lines',x=data_revenue['event_date'], 
                                              y=data_revenue['tiktok-ads'], 
                                              name=f'Затраты на рекламу tiktok-ads'))
fig_revenue.add_trace(go.Scatter(mode='lines',x=data_revenue['event_date'], 
                                              y=data_revenue['pinterest-ads'], 
                                              name=f'Затраты на рекламу pinterest-ads'))
fig_revenue.add_trace(go.Scatter(mode='lines',x=data_revenue['event_date'], 
                                              y=data_revenue['snapchat-ads'], 
                                              name=f'Затраты на рекламу snapchat-ads'))
fig_revenue.update_layout(title=f'Выручка и затраты на рекламу, {selected_shop}')

fig_mean_path = px.bar(x=mean_time.keys(), y=mean_time.values(), title='Среднее время между покупкой и первым контактом с каналом',
                                 labels={'x': 'Канал', 'y': 'Время (дни)'})

# Display the plots side by side
st.plotly_chart(fig)
# col1, col2 = st.columns(2)
st.plotly_chart(fig_channels)
st.plotly_chart(fig_prob)
selected_revenue = st.selectbox('Показать результат MMM модели?', ['нет', 'да'])
if selected_revenue == 'да':
    lr, X, y = train_mmm_model(data_revenue.drop(columns='spendings'), selected_shop)
    fig = draw_mmm_result(lr, X, y)
    st.plotly_chart(fig)
    pass
else:
    st.plotly_chart(fig_revenue)
st.plotly_chart(fig_mean_path)