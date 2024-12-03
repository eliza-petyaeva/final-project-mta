import pandas as pd
import ast
from typing import List, Dict, Tuple

def prep_data_for_markov_shapley(data, mta_level:  str='source') -> Tuple[Dict[str, Tuple[int, pd.DataFrame]], Dict[str, float]]:
    """
    Args:
        mta_level (str):            - source or adid
    Returns:
        mta_data (pd.DataFrame):    - data for mta
    """
    budget = {}
    # make a string path with > delimeter. Ex: facebook>google>facebook
    data[f'path'] = data[f'tw_{mta_level}'].apply(lambda x: '>'.join(x))
    # for each shop from dataframe prepare dataset
    mta_data_shops = {}
    for shop in data.shop_name.unique():
        total_conv = {}
        data_shop = data[data.shop_name == shop]
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
        budget[f'{shop}'] = mta_data['total_conversion_value'].sum()

    return mta_data_shops, budget

def prepare_data(data:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    names_sources = open('data/names_sources.txt', "r")
    names_sources = names_sources.read()
    names_sources = list(map(lambda x: x.split(','),
                                                    names_sources.replace("'", '')
                                                                    .replace("]", '')
                                                                    .replace("[", '')
                                                                    .replace('"', '')
                                                                    .replace(' ', '')
                                                                    .split(';')))

    FB_LIST, GOOGLE_LIST, TIKTOK_LIST, SNAPCHAT_LIST, PINTEREST_LIST, KLAVIYO_LIST, EMAIL_LIST, INFLUENCER_LIST = names_sources

    KLAVIYO_LIST.append('kl')

    def clean_data(path: List[str]) -> List[str]:
        """
        Cleans customer journey paths.

        Args:
            path (list[str]): Customer journey path (only sources).

        Returns:
            list[str]: Cleaned customer journey path.
        """
        new_path = []
        for source in path:
            source = source.lower().replace("\n", '').replace("(", '').replace(")", '')
            source = source.replace("[", '').replace("]", '').replace("'", '')
            if source in FB_LIST:
                source = 'facebook'
            elif source in GOOGLE_LIST:
                source = 'google'
            elif source in TIKTOK_LIST:
                source = 'tiktok'
            elif source in SNAPCHAT_LIST:
                source = 'snapchat'
            elif source in PINTEREST_LIST:
                source = 'pinterest'
            elif source in KLAVIYO_LIST:
                source = 'klaviyo'
            elif source in EMAIL_LIST:
                source = 'email'
            elif source in INFLUENCER_LIST:
                source = 'influencer'
            elif source == '':
                source = 'unset'
            new_path.append(source)
        return new_path

    # data['tw_source'] = data['tw_source'].apply(lambda x: ast.literal_eval(x))

    data_shop_1 = data[data['shop_name'] == 'beauty_shop_1']
    data_shop_2 = data[data['shop_name'] == 'beauty_shop_2']

    data_shop_1['tw_source_clean'] = data_shop_1['tw_source'].apply(clean_data)
    data_shop_2['tw_source_clean'] = data_shop_2['tw_source'].apply(clean_data)

    data_shop_1['total_price'] = data_shop_1['total_price'].astype(float)
    data_shop_2['total_price'] = data_shop_2['total_price'].astype(float)

    return data_shop_1, data_shop_2


def sort_most_popular_platforms(count_all_platforms: Dict[str, int], n: int) -> Dict[str, int]:
    """
    Sorts platforms by popularity.

    Args:
        count_all_platforms (dict): Dictionary of platforms and their counts.
        n (int): Number of top platforms to return.

    Returns:
        dict: Top n platforms sorted by count.
    """
    return dict(sorted(count_all_platforms.items(), key=lambda x: x[1], reverse=True)[:n])

def calc_total_price(data: pd.DataFrame, shop: str, date_start: str, date_end: str) -> float:
    """
    Args:
        data (pd.DataFrame): - dataframe with all orders
        shop (str):          - shop name
        date (str):          - date
    Returns:
        total_price (float): - total price of all orders
    """

    return data[
            (data['shop_name'] == shop) &
            (data['journey_end_ts'] >= date_start) &
            (data['journey_end_ts'] < date_end)
        ]['total_price'].sum()


def sort_attribution_result(attribution_result: Dict[str, float]) -> Dict[str, float]:
    """
    Args:
        attribution_result (dict): - dict with attribution result
    Returns:
        sorted_attribution_result (dict): - dict with sorted attribution result
    """

    return dict(sorted(attribution_result.items(), key=lambda x: x[1], reverse=True))
