from __future__ import annotations

import pandas as pd
import typing
import numpy as np
import config
# type: ignore
import mta_algorithms as mta_
from google.cloud import bigquery
from datetime import timedelta
import ast

names_sources_path = "data/names_sources.txt"

class Mta_Conversion():
    def __init__(self, shop) -> None:

        """
        Args:
            shops (list):  - shop name
        """
        self.shop:   list           = shop
        self.data:   pd.DataFrame   = pd.DataFrame()
        self.budget: dict           = {}
        self.adid_source_dict: dict = {}

        self.names_sources = open(names_sources_path, "r")
        self.names_sources = self.names_sources.read()
        self.names_sources = list(map(lambda x: x.split(','),
                                                self.names_sources.replace("'", '')
                                                                  .replace("]", '')
                                                                  .replace("[", '')
                                                                  .replace('"', '')
                                                                  .replace(' ', '')
                                                                  .split(';')))

        self.FB_LIST, self.GOOGLE_LIST, self.TIKTOK_LIST, self.SNAPCHAT_LIST, self.PINTEREST_LIST, self.KLAVIYO_LIST, self.EMAIL_LIST, self.INFLUENCER_LIST = self.names_sources

    def clean_data(self, path:list) -> list:
        """
        Args:
            path (list[str]):     - customer journey path (only sources)
        Returns:
            new_path (list[str]): - clean customer journey path
        """
        new_path = []

        for source in path:
            source = source.lower()
            source = source.replace("\n", '')
            source = source.replace("(", '')
            source = source.replace(")", '')
            source = source.replace("[", '')
            source = source.replace("]", '')
            source = source.replace("'", '')
            if source in self.FB_LIST:
                source = 'facebook'
            elif source in self.GOOGLE_LIST:
                source = 'google'
            elif source in self.TIKTOK_LIST:
                source = 'tiktok'
            elif source in self.SNAPCHAT_LIST:
                source = 'snapchat'
            elif source in self.PINTEREST_LIST:
                source = 'pinterest'
            elif source in self.KLAVIYO_LIST:
                source = 'klavio'
            elif source in self.EMAIL_LIST:
                source = 'email'
            elif source in self.INFLUENCER_LIST:
                source = 'influencer'
            elif source == '':
                source = 'unset'
            # else:
            #     source = 'other'
            new_path.append(source)

        return new_path

    def prep_data(self, mta_level:  str='adid') -> dict:
        """
        Args:
            mta_level (str):            - source or adid
        Returns:
            mta_data (pd.DataFrame):    - data for mta
        """
        # make a string path with > delimeter. Ex: facebook>google>facebook
        self.data[f'path'] = self.data[f'tw_{mta_level}'].apply(lambda x: '>'.join(x))
        # for each shop from dataframe prepare dataset
        mta_data_shops = {}
        for shop in self.data.shop_name.unique():
            total_conv = {}
            data_shop = self.data[self.data.shop_name == shop]
            # for each unique path calculate its conversion and lost customers
            for el in data_shop['path'].unique():
                data_path = data_shop[data_shop['path'] == el]
                tot_conv = data_path[data_path.journey_success == 1].shape[0]
                total_conversion_value = data_path[data_path.journey_success == 1].total_price.sum()
                tot_null = data_path[data_path.journey_success == 0].shape[0]
                total_conv[el] = {'total_conversions': tot_conv,
                                  'total_conversion_value': total_conversion_value,
                                  'total_null': tot_null}
            mta_data = pd.DataFrame(total_conv).T.reset_index().rename(columns={'index':'path'})
            mta_data_shops[f'{shop}'] = mta_data.shape[0], mta_data
            # calculate total budget for each shop
            self.budget[f'{shop}'] = mta_data['total_conversion_value'].sum()

        return mta_data_shops

    def calc_mta(self, mta_data: pd.DataFrame) -> object:
        """
        Args:
            mta_data (pd.DataFrame):   - data for mta
        Returns:
            mta (object):              - mta object with calculated coefs
        """
        # calculate mta with 2 algorithms
        mta = mta_.MTA(mta_data)
        mta.markov()
        mta.shapley()

        return mta

    def prep_data_clean_adid(self):
        """
        Returns:
            adid_source_dict (dict): - pairs (shop_adid : source) for output
        """
        # delete data with null paths (yes, we can lost money, but it's not useful)
        data_clean = self.data[self.data.tw_adid.astype(str) != self.data.tw_source.astype(str)]

        # create source-shop-adid dict for output
        data_clean['shop_name_for_flatten'] = data_clean['shop_name'].apply(lambda x: [x]) * data_clean['len_tw_source']
        ads_flatten = [ad for path in data_clean['tw_adid'].values for ad in path]
        ss_flatten = [ad for path in data_clean['tw_source'].values for ad in path]
        shop_flatten = [shop for path in data_clean['shop_name_for_flatten'].values for shop in path]
        shop_ads_id = [shop_flatten[i].replace(' ', '') + ads_flatten[i].replace(' ', '') for i in range(len(shop_flatten))]
        self.adid_source_dict = dict(zip(shop_ads_id, ss_flatten))
        return

    def save_data(self, mta_data: dict, mta_level: str) -> list:
        """
        Args:
            mta_data (pd.DataFrame): - data for mta
            mta_level (str):         - source or adid
        Returns:
            mta_result (list):       - list with coef and conversion
        """
        mta_result = []

        for shop_name in mta_data:
            shop_res = {}
            shop_res['shop'] = shop_name
            models_results = [{'model':'markov', 'influence_percent':[], 'conversion':[]},
                              {'model':'shapley', 'influence_percent':[], 'conversion':[]}
                                ]

            # errors and empty dataframes check
            if mta_data[shop_name][0] == 0:
                shop_res['data'] = models_results
                mta_result.append(shop_res)
                continue
            try:
                # mta calculation
                mta = self.calc_mta(mta_data[shop_name][1])
            except ZeroDivisionError:
                shop_res['data'] = models_results
                mta_result.append(shop_res)
                continue

            # normalization
            values_markov = list(mta.attribution['markov'].values()) / np.sum(list(mta.attribution['markov'].values()))
            values_shapley = list(mta.attribution['shapley'].values()) / np.sum(list(mta.attribution['shapley'].values()))
            channels = mta.channels

            if mta_level == 'source':
                for el in range(len(values_markov)):
                    # add markov
                    models_results[0]['influence_percent'].append(
                                                {'source': channels[el],
                                                 'value':values_markov[el]})
                    models_results[0]['conversion'].append(
                                                {'source': channels[el],
                                                 'value':values_markov[el] * self.budget[shop_name]})
                    # add shapley
                    try: # if markov != 0 but Shapley = 0, we add zero lists
                        models_results[1]['influence_percent'].append(
                                                    {'source': channels[el],
                                                    'value':values_shapley[el]})
                        models_results[1]['conversion'].append(
                                                    {'source': channels[el],
                                                    'value':values_shapley[el] * self.budget[shop_name]})
                    except IndexError:
                        models_results = [{'model':'markov', 'influence_percent':[], 'conversion':[]},
                                          {'model':'shapley', 'influence_percent':[], 'conversion':[]}
                                         ]

            elif mta_level == 'adid':
                for el in range(len(values_markov)):

                    source_adid = self.adid_source_dict[shop_name.replace(' ', '')+channels[el].replace(' ', '')] if channels[el] != shop_name else 'unset'

                    # add markov. source get by key 'shop_name+adid'
                    models_results[0]['influence_percent'].append(
                                                {'adid': channels[el],
                                                 'source': source_adid,
                                                 'value':values_markov[el]})
                    models_results[0]['conversion'].append(
                                                {'adid': channels[el],
                                                 'source': source_adid,
                                                 'value':values_markov[el] * self.budget[shop_name]})
                    # add shapley
                    try: # if markov != 0 but Shapley = 0, we add zero lists
                        models_results[1]['influence_percent'].append(
                                                    {'adid': channels[el],
                                                    'source': source_adid,
                                                    'value':values_shapley[el]})
                        models_results[1]['conversion'].append(
                                                    {'adid': channels[el],
                                                    'source': source_adid,
                                                    'value':values_shapley[el] * self.budget[shop_name]})
                    except IndexError:
                        models_results = [{'model':'markov', 'influence_percent':[], 'conversion':[]},
                                          {'model':'shapley', 'influence_percent':[], 'conversion':[]}
                                         ]
            shop_res['data'] = models_results
            mta_result.append(shop_res)
        return mta_result

    def order_output(self, mta_result, hour=1) -> list:
        """
        Args:
            mta_result (list):       - list with coef and conversionv
            hour (int):              - period forlast  orders
        Returns:
            result     (list):       - orders list with coef and conversion
        """
        result = []
        data_hour = self.data[(self.data.journey_end_ts >= self.data.journey_end_ts.max() - timedelta(hours=hour)) & (self.data.journey_success == 1)]
        for el in mta_result:
            # create dictionaries with ad names and values
            data_shop = data_hour[(data_hour.shop_name == el['shop']) ]
            markov_percent = dict(zip([x['adid'] for x in el['data'][0]['influence_percent']],
                                      [x['value'] for x in el['data'][0]['influence_percent']]))

            shapley_percent = dict(zip([x['adid'] for x in el['data'][1]['influence_percent']],
                                       [x['value'] for x in el['data'][1]['influence_percent']]))

            for order_ind in data_shop.index:
                order = data_shop.loc[order_ind]
                path_source = order['tw_source']
                path_adid = order['tw_adid']
                attribution = [
                    {
                        "source": path_source[x],
                        "ad_id": path_adid[x],
                        "markov": {
                                "influence_percent": markov_percent[path_adid[x]],
                                "conversion": markov_percent[path_adid[x]] * data_shop['total_price'][order_ind],
                            },
                        "shapley": {
                                "influence_percent": shapley_percent[path_adid[x]],
                                "conversion":shapley_percent[path_adid[x]] * data_shop['total_price'][order_ind],
                            }
                    }
                    for x in range(len(path_source)) ]

                result.append(
                    {
                        "order_id": order['order_id'],
                        "shop": el['shop'],
                        "attribution": attribution

                    }
                )
        return result

    def mta_conversion(self,
                       mta_level:   str='adid') -> tuple:
        """
        Args:
            mta_level (str):      - source or adid
        Returns:
            mta_result (list):      - list with coef and conversion
            mta_order_result (list) - orders list with coef and conversion
        """

        mta_result = []
        self.get_data(mta_level)
        self.data.reset_index(drop=True, inplace=True)
        self.data['tw_source'] = self.data['tw_source'].apply(lambda x: ast.literal_eval(x))
        self.data['tw_source'] = self.data['tw_source'].apply(lambda x: self.clean_data(x))

        if mta_level == 'adid':
            self.prep_data_clean_adid()

        self.data['total_price'] = self.data['total_price'].astype(float)
        mta_data = self.prep_data(mta_level)
        mta_result = self.save_data(mta_data, mta_level)
        if mta_level == 'adid':
            mta_order_result = self.order_output(mta_result)
            return mta_result, mta_order_result
        else:
            return mta_result, []

if __name__ == "__main__":


    query_mta = pd.read_json('mta/example_query.json')
    shop = list(query_mta['shop'])
    mta_conv = Mta_Conversion(shop=shop)
    mta_conv_result = mta_conv.mta_conversion(mta_level=query_mta['mta_level'][0])

    print(mta_conv_result)
