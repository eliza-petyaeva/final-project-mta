import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
import mta_algorithms as mta_


def last_touch_attribution(data: pd.DataFrame, shop: str) -> Dict[str, float]:
    lta_result = {}
    data_shop = data[(data.shop_name == shop)&(data.total_price > 0)]
    for i in data_shop.index:
        if data_shop['tw_source_clean'][i][-1] in lta_result:
            lta_result[data_shop['tw_source_clean'][i][-1]] += data_shop['total_price'][i]
        else:
            lta_result[data_shop['tw_source_clean'][i][-1]] = data_shop['total_price'][i]
    
    return lta_result

def first_touch_attribution(data: pd.DataFrame, shop: str) -> Dict[str, float]:
    fta_result = {}
    data_shop = data[(data.shop_name == shop)&(data.total_price > 0)]
    for i in data_shop.index:
        if data_shop['tw_source_clean'][i][0] in fta_result:
            fta_result[data_shop['tw_source_clean'][i][0]] += data_shop['total_price'][i]
        else:
            fta_result[data_shop['tw_source_clean'][i][0]] = data_shop['total_price'][i]
    return fta_result

def linear_attribution(data: pd.DataFrame, shop: str) -> Dict[str, float]:
    linear_result = {}
    data_shop = data[(data.shop_name == shop)&(data.total_price > 0)]
    for i in data_shop.index:
        for j in data_shop['tw_source_clean'][i]:
            if j in linear_result:
                linear_result[j] += data_shop['total_price'][i]/len(data_shop['tw_source_clean'][i])
            else:
                linear_result[j] = data_shop['total_price'][i]/len(data_shop['tw_source_clean'][i])
    return linear_result


def time_decay_attribution(data: pd.DataFrame, shop: str) -> Dict[str, float]:
    time_decay_result = {}
    data_shop = data[(data.shop_name == shop)&(data.total_price > 0)]
    for i in data_shop.index:
        weights = np.linspace(0, 1, len(data_shop['tw_source_clean'][i]) + 1) / sum(np.linspace(0, 1, len(data_shop['tw_source_clean'][i]) + 1))
        for j in range(len(data_shop['tw_source_clean'][i])):
            if data_shop['tw_source_clean'][i][j] in time_decay_result:
                time_decay_result[data_shop['tw_source_clean'][i][j]] += data_shop['total_price'][i]*weights[j + 1]
            else:
                time_decay_result[data_shop['tw_source_clean'][i][j]] = data_shop['total_price'][i]*weights[j + 1]
    return time_decay_result

def position_based_attribution(data: pd.DataFrame, shop: str) -> Dict[str, float]:
    position_based_result = {}
    data_shop = data[(data.shop_name == shop)&(data.total_price > 0)]
    for i in data_shop.index:
        for j in range(len(data_shop['tw_source_clean'][i])):
            if j == 0 or j == len(data_shop['tw_source_clean'][i])-1:
                if data_shop['tw_source_clean'][i][j] in position_based_result:
                    position_based_result[data_shop['tw_source_clean'][i][j]] += data_shop['total_price'][i]* 0.4
                else:
                    position_based_result[data_shop['tw_source_clean'][i][j]] = data_shop['total_price'][i]* 0.4
            else:
                if data_shop['tw_source_clean'][i][j] in position_based_result:
                    position_based_result[data_shop['tw_source_clean'][i][j]] += data_shop['total_price'][i]/(len(data_shop['tw_source_clean'][i])-2) * 0.2
                else:
                    position_based_result[data_shop['tw_source_clean'][i][j]] = data_shop['total_price'][i]/(len(data_shop['tw_source_clean'][i])-2) * 0.2
    return position_based_result

def markov_attribution(mta_data, budget):

    mta_result = []
    for shop_name in mta_data:
        shop_res = {}
        shop_res['shop'] = shop_name
        models_results = {'model':'markov', 'influence_percent':{}, 'conversion':{}}
                            
        # errors and empty dataframes check
        if mta_data[shop_name][0] == 0:
            shop_res['data'] = models_results
            mta_result.append(shop_res)
            continue
        try:
            # mta calculation
            mta = mta_.MTA(mta_data[shop_name][1])
            mta.markov()
        except ZeroDivisionError:
            shop_res['data'] = models_results
            mta_result.append(shop_res)
            continue
        values_markov = list(mta.attribution['markov'].values()) / np.sum(list(mta.attribution['markov'].values()))
        channels = mta.channels
        for el in range(len(values_markov)):
            # add markov
            models_results['influence_percent'][channels[el]] = values_markov[el]
            models_results['conversion'][channels[el]] = values_markov[el] * budget[shop_name]

        shop_res['data'] = models_results
        mta_result.append(shop_res)

    return mta_result

def shapley_attribution(mta_data: Dict[str, Tuple[int, Any]], budget: Dict[str, float]) -> List[Dict[str, Any]]:

    mta_result = []
    for shop_name in mta_data:
        shop_res = {}
        shop_res['shop'] = shop_name
        models_results = {'model':'shapley', 'influence_percent':{}, 'conversion':{}}

        # errors and empty dataframes check
        if mta_data[shop_name][0] == 0:
            shop_res['data'] = models_results
            mta_result.append(shop_res)
            continue
        try:
            # mta calculation
            mta = mta_.MTA(mta_data[shop_name][1])
            mta.shapley()
        except ZeroDivisionError:
            shop_res['data'] = models_results
            mta_result.append(shop_res)
            continue
        values_shapley = list(mta.attribution['shapley'].values()) / np.sum(list(mta.attribution['shapley'].values()))
        channels = mta.channels
        for el in range(len(values_shapley)):
            # add shapley
            models_results['influence_percent'][channels[el]] = values_shapley[el]
            models_results['conversion'][channels[el]] = values_shapley[el] * budget[shop_name]

        shop_res['data'] = models_results
        mta_result.append(shop_res)
    return mta_result

def mean_channel_attribution_time(data: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
    data = data[data.journey_success == 1]
    data_dic_timing = {}
    all_channels = []
    for i in data.index:
        used_sources = []
        for j in range(len(data['tw_source_clean'][i])):
            if data['tw_source_clean'][i][j] not in used_sources:
                if data['tw_source_clean'][i][j] in data_dic_timing:
                    data_dic_timing[data['tw_source_clean'][i][j]] += data['time_between_order_and_step'][i][j]
                    data_dic_timing[f"{data['tw_source_clean'][i][j]}_place"] += j / len(data['tw_source_clean'][i])
                    data_dic_timing[f"{data['tw_source_clean'][i][j]}_times"] += 1
                else:
                    data_dic_timing[data['tw_source_clean'][i][j]] = data['time_between_order_and_step'][i][j]
                    data_dic_timing[f"{data['tw_source_clean'][i][j]}_place"] = j / len(data['tw_source_clean'][i])
                    data_dic_timing[f"{data['tw_source_clean'][i][j]}_times"] = 1
                    all_channels.append(data['tw_source_clean'][i][j])
                used_sources.append(data['tw_source_clean'][i][j])
    for channel in all_channels:
        data_dic_timing[channel] = data_dic_timing[channel]/data_dic_timing[f"{channel}_times"]
        data_dic_timing[f"{channel}_place"] = data_dic_timing[f"{channel}_place"]/data_dic_timing[f"{channel}_times"]
    return all_channels, data_dic_timing

def sorted_mean_channel_attribution_time(mcat: Tuple[List[str], Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    mean_time = {}
    mean_place = {}
    times = {}
    for element in mcat[0]:
        mean_time[element] = mcat[1][element] / (60 * 24)
        mean_place[element] = mcat[1][f"{element}_place"]
        times[element] = mcat[1][f"{element}_times"]
    
    mean_time = {k: v for k, v in sorted(mean_time.items(), key=lambda item: item[1], reverse=True)}
    mean_place = {k: v for k, v in sorted(mean_place.items(), key=lambda item: item[1], reverse=True)}
    times = {k: v for k, v in sorted(times.items(), key=lambda item: item[1], reverse=True)}

    return mean_time, mean_place, times

