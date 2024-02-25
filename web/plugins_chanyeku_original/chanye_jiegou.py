import pickle
import requests
import json
import pandas as pd
import random
import os 
#from .location_mapper import get_code_from_str
from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine
#from location_mapper import get_code_from_str
conf = {
    "user": "default",
    "password": "",
    "server_host": "10.0.0.25",
    "port": "8123",
    "db": "algorithm"
}
def get_ck_data(sql):
    connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
    engine = create_engine(connection, pool_size=100, pool_recycle=3600, pool_timeout=20)
    
    #sql = text('SHOW TABLES')
    sql = text(sql)
    
    session = make_session(engine)
    data = []
    cursor = session.execute(sql)
    try:
        fields = cursor._metadata.keys
        df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
        data = df
    finally:
        cursor.close()
        session.close()
    return data
ALL_CY = ['软件和信息服务','量子信息','物联网','5G','医疗器械','智能网联汽车','新能源汽车','机器人','生物医药','纺织服装','建材','半导体','区块链','工业互联网','工业母机','海洋','新材料','新能源','金融科技','人工智能','智能家居','工程机械','石油化工','冶金','数字创意','安全应急','检验检测','节能环保','煤化工','航空航天','激光与增材制造','烟草','轨道交通','元宇宙','农业与食品','超高清视频显示']
def get_chanyejiegou(city_name,chanye=''):
    youshi = {}
    lieshi = {}
    mid = {}
    sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `城市` like '%{city_name}%';"
    data = get_ck_data(sql)
    chanye_rank = {}
    for _,da in data.iterrows():
        cy = da['产业']
        paiming = da['排名']
        if cy not in chanye_rank:
            chanye_rank[cy] = []
        chanye_rank[cy].append(paiming)
    for cy in chanye_rank:
        chanye_rank[cy]=int(sum(chanye_rank[cy])/len(chanye_rank[cy]))
    for cy in chanye_rank:
        #paiming,status = get_chanye_status(city_name,cy)
        paiming = chanye_rank[cy]
        if cy == '全部产业链':
            continue
        #if not paiming:
        #    continue
        if paiming <= 30:
            youshi[cy]=paiming
            continue
        if chanye_rank[cy] > 60:
            lieshi[cy]=paiming
            continue
        mid[cy]=paiming
    youshi_sorted = sorted(youshi.items(),key=lambda k:k[1])
    youshi_cy = [k for k,v in youshi_sorted]
    mid_sorted = sorted(mid.items(),key=lambda k:k[1])
    mid_cy = [k for k,v in mid_sorted]
    lieshi_sorted = sorted(lieshi.items(),key=lambda k:k[1])
    lieshi_cy = [k for k,v in lieshi_sorted]
    youshi_paiming = ['{}在全国的排名是第{}位'.format(cy,youshi[cy]) for cy in youshi_cy]
    mid_paiming = ['{}在全国的排名是第{}位'.format(cy,mid[cy]) for cy in mid_cy]
    lieshi_paiming = ['{}在全国的排名是第{}位'.format(cy,lieshi[cy]) for cy in lieshi_cy]
    all_chanye = '、'.join(youshi_cy+mid_cy + lieshi_cy)
    cy_paiming_str = '\n'.join(youshi_paiming + mid_paiming + lieshi_paiming)
    youshi_str = '、'.join(youshi_cy) + '属于优势产业链。'
    mid_str = '、'.join(mid_cy) + '属于居中产业链。'
    lieshi_str = '、'.join(lieshi_cy) + '属于劣势产业链。'
    if not youshi_cy:
        youshi_str = ''
    if not mid_cy:
        mid_str = ''
    if not lieshi_cy:
        lieshi_str = ''
    answer = f"""{city_name}当前存在的主要产业共有{len(youshi_cy) + len(lieshi_cy) + len(mid_cy)}条，分别是{all_chanye}。
    其中，从产业链企业规模看、从产业链重点企业规模看、从产业链发展趋势看：
    {cy_paiming_str}
    {youshi_str}\n\n{mid_str}\n{lieshi_str}"
    """
    if not paiming:
        answer = ''
    return answer
if __name__ == '__main__':
    city_name='烟台'
    #city_name='北京'
    chanye='新能源'
    data = get_chanyejiegou(city_name,chanye='新能源')
    #data = get_chanyepaiming(city_name='北京',chanye='新能源')
    print(data)
