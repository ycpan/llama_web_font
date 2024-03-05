import pickle
import requests
import json
import pandas as pd
import random
import os 
from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine
#from .location_mapper import get_code_from_str
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
def get_chanyepaiming(city_name,chanye=''):
    #paiming,status = get_chanye_status(city_name,chanye)
    sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `城市` like '%{city_name}%' and (`产业` like '%{chanye}%' or `产业节点` like '%{chanye}%');"
    data = get_ck_data(sql)
    if  data.empty:
        return ""
    chanye_rank = {}
    for _,da in data.iterrows():
        cy = da['产业']
        paiming = da['排名']
        if cy not in chanye_rank:
            chanye_rank[cy] = []
        chanye_rank[cy].append(paiming)
    for cy in chanye_rank:
        chanye_rank[cy]=int(sum(chanye_rank[cy])/len(chanye_rank[cy]))
    #paiming = chanye_rank[chanye]
    paiming = int(sum(chanye_rank.values()) / len(chanye_rank.values()))
    status = '非优势产业'
    if paiming <= 30:
        status = '优势产业'

    answer = f'{chanye}在{city_name}是{status},全国排名第{paiming}位'
    if not paiming:
        answer = ''
    return answer
if __name__ == '__main__':
    city_name='烟台'
    chanye='新能源'
    #data = get_chanyepaiming(city_name='烟台',chanye='新能源')
    data = get_chanyepaiming(city_name='北京',chanye='新能源')
    #data = get_chanyepaiming(city_name='北京',chanye='行业应用')
    #data = get_chanyepaiming('林州市','电力应急保障产品')
    print(data)
