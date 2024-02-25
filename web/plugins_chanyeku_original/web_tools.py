# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 14:32
# @Author  : lkl
# @File    : weather_api.py
# @Description :
import requests
import pandas as pd
def get_city_code_dict():
    res = {}
    #df = pd.read_excel('com_weather_city.xlsx')
    df = pd.read_excel('/devdata/home/user/panyongcan/Project/chatweb1/dev/main/llama_web_font/web/plugins_chanyeku/com_weather_city.xlsx')
    for idx,da in df.iterrows():
        city = da['level3']
        code = da['areaid']
        res[city] = code
    return res

city_code = get_city_code_dict()
def get_city_code(city):
    city = city.replace('市','').replace('区','').replace('县','')
    if city in city_code:
        return city_code[city]
    return ''


def get_weather(city, level2="", level3=""):
    """
        实时新闻接口
    :param city_code:
    :param level2:
    :param level3:
    :return:
    """
    url = "http://10.0.0.15:19998/com_weather/realtime"

    city_code = get_city_code(city)
    if not city_code:
        return ''
    params = {
        "city_code": city_code,
        "level2": level2,
        "level3": level3
    }

    data = requests.get(url, params=params).json()
    data = data['data']
    "{'code': 0, 'data': {'nameen': 'yantai', 'cityname': '烟台', 'city': '101120501', 'temp': '3', 'tempf': '37.4', 'WD': '东风', 'wde': 'E', 'WS': '1级', 'wse': '5km/h', 'SD': '63%', 'sd': '63%', 'qy': '1021', 'njd': '27km', 'time': '13:55', 'rain': '0', 'rain24h': '0', 'aqi': '30', 'aqi_pm25': '30', 'weather': '晴', 'weathere': 'Sunny', 'weathercode': 'd00', 'limitnumber': '', 'date': '12月27日(星期三)'}}"
    #import ipdb
    #ipdb.set_trace()
    s = f"今天是{data['date']},{data['cityname']}天气{data['weather']},降雨概率{data['rain']},{data['WD']}{data['WS']},温度{data['temp']},湿度{data['sd']}"
    return s


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
    data = requests.get(url, params=params).json()
    "{'title': '半导体及元件板块持续走高 金海通涨停', 'url': 'http://www.eeo.com.cn/2023/1227/621697.shtml', 'pub_date': '2023-12-27', 'summary': '半导体及元件板块 持续走高,金海通涨停,芯海科技涨超10%,聚辰股份涨超8%,恒烁股份、中微公司、拓荆科技 、蓝箭电子、北方华创等多股涨超5%。 分享  收藏 热新闻 2024研招报告:“双非”大学报 名人数继续增  【经观讲堂第32期】李嘉...', 'source': '经济观察网'} "
    res = []
    for da in data:
        #sub_str = f"{da['title']}。{da['summary']}。来源{da['source']},发布时间{da['pub_date']},{da['url']}"
        sub_str = f"{da['summary']}来源{da['source']},发布时间{da['pub_date']},{da['url']}"
        res.append(sub_str)
    return "\n".join(res)



if __name__ == '__main__':
    #weather = get_weather(city_code="101042600")
    #weather = get_weather('烟台')
    #print(weather)
    baidu_news = get_baidu_news("半导体新闻")
    print(baidu_news)
