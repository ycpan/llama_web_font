# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 14:32
# @Author  : lkl
# @File    : weather_api.py
# @Description :
import requests


def get_weather(*, city_code="", level2="", level3=""):
    """
        实时新闻接口
    :param city_code:
    :param level2:
    :param level3:
    :return:
    """
    url = "http://10.0.0.15:19998/com_weather/realtime"
    params = {
        "city_code": city_code,
        "level2": level2,
        "level3": level3
    }

    return requests.get(url, params=params).json()


def get_baidu_news(wd):
    """
        百度资讯接口
    :param wd:
    :return:
    """
    url = "http://10.0.0.15:19998/baidu_news/news20"

    params = {
        "wd": wd
    }
    return requests.get(url, params=params).json()


if __name__ == '__main__':
    weather = get_weather(city_code="101042600")
    print(weather)
    baidu_news = get_baidu_news("半导体新闻")
    print(baidu_news)
