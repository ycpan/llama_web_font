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
def get_chanye2code():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    index  = dir_path.find('/web')
    web_path = dir_path[0:index + 5]
    csv_path = os.path.join(web_path,'plugins_chanyeku/chanye2code.csv')
    data = pd.read_csv(csv_path)
    return data
payload = ""
headers = {}
#f = open('/devdata/home/user/panyongcan/Project/llama_web_font/web/plugins_chanyeku/industry2code.pkl','rb')
#industry2code = pickle.load(f)
#industrycode = {}
#for key in industry2code:
#    for sub_key in industry2code[key]:
#        industrycode[sub_key]=industry2code[key][sub_key]
#        industrycode[sub_key.replace('产业','')]=industry2code[key][sub_key]
ALL_CY = ['软件和信息服务','量子信息','物联网','5G','医疗器械','智能网联汽车','新能源汽车','机器人','生物医药','纺织服装','建材','半导体','区块链','工业互联网','工业母机','海洋','新材料','新能源','金融科技','人工智能','智能家居','工程机械','石油化工','冶金','数字创意','安全应急','检验检测','节能环保','煤化工','航空航天','激光与增材制造','烟草','轨道交通','元宇宙','农业与食品','超高清视频显示']
def get_chanyeruoshi(city_name,chanye=''):
    youshi = {}
    lieshi = {}
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

    lieshi = []
    for cy in ALL_CY:
        if cy not in chanye_rank or chanye_rank[cy] > 60:
            lieshi.append(cy)
    #youshi = {}
    #for cy in chanye_rank:
    #    if cy == '全部产业链':
    #        continue
    #    #paiming,status = get_chanye_status(city_name,cy)
    #    paiming = chanye_rank[cy]
    #    #if not paiming:
    #    #    continue
    #    if paiming < 50:
    #        continue
    #    #if status == '弱势产业':
    #    youshi[cy]=paiming
    #    #else:
    #    #    lieshi[cy]=paiming
    #youshi_sorted = sorted(youshi.items(),key=lambda k:k[1])
    #youshi_cy = [k for k,v in youshi_sorted]
    ##lieshi_sorted = sorted(lieshi.items(),key=lambda k:k[1])
    ##lieshi_cy = [k for k,v in lieshi_sorted]
    #youshi_paiming = ['{}在全国的排名是第{}位'.format(cy,youshi[cy]) for cy in youshi_cy]
    ##lieshi_paiming = ['{}在全国的排名是第{}位'.format(cy,lieshi[cy]) for cy in lieshi_cy]
    ##all_chanye = '、'.join(youshi_cy+lieshi_cy)
    #all_chanye = '、'.join(youshi_cy)
    ##cy_paiming_str = '\n'.join(youshi_paiming + lieshi_paiming)
    #cy_paiming_str = '\n'.join(youshi_paiming)
    answer = f"""{city_name}当前存在的弱势产业共有{len(lieshi)}条，分别是{','.join(lieshi)}。"""
    if not paiming:
        answer = ''
    return answer
if __name__ == '__main__':
    city_name='烟台'
    chanye='新能源'
    data = get_chanyeruoshi(city_name='烟台',chanye='新能源')
    #data = get_chanyepaiming(city_name='北京',chanye='新能源')
    print(data)
