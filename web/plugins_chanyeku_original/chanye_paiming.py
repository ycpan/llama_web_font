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
#def get_chanye2code():
#    dir_path = os.path.dirname(os.path.realpath(__file__))
#    index  = dir_path.find('/web')
#    web_path = dir_path[0:index + 5]
#    csv_path = os.path.join(web_path,'plugins_chanyeku/chanye2code.csv')
#    data = pd.read_csv(csv_path)
#    return data
#
#payload = ""
#headers = {}
##f = open('/devdata/home/user/panyongcan/Project/llama_web_font/web/plugins_chanyeku/industry2code.pkl','rb')
##industry2code = pickle.load(f)
##industrycode = {}
##for key in industry2code:
##    for sub_key in industry2code[key]:
##        industrycode[sub_key]=industry2code[key][sub_key]
##        industrycode[sub_key.replace('产业','')]=industry2code[key][sub_key]
#data = get_chanye2code()
#industrycode = {}
#for _,da in data.iterrows():
#    code = da['node']
#    chanye = da['name']
#    industrycode[chanye]=code
#    chanye = chanye.replace('产业','')
#    industrycode[chanye]=code
#
#def get_chanye_status(city_name,chanye=''):
#    #url = "https://devdly.incostar.com/api/special/industrial/end/getIndustrialStatus?nodeCode=H03&areaCode=110100"
#    if chanye and chanye in industrycode:
#        node_code=industrycode[chanye]
#    else:
#        node_code = ''
#    area_code = get_code_from_str(city_name)
#    area_code=area_code.replace('0000','0100')
#    url = f"https://devdly.incostar.com/api/special/industrial/end/getIndustrialStatus?nodeCode={node_code}&areaCode={area_code}"
#    response = requests.request("GET", url, headers=headers, data=payload)
#    data = json.loads(response.text)
#    if data['code'] == 500:
#        paiming = random.randint(50,100)
#        status = '非优势产业'
#        return paiming,status
#    paiming = data['data']['rank']
#    status = '优势产业' if data['data']['status'] == '1' else '非优势产业'
#    return paiming,status
def get_chanyepaiming(city_name,chanye=''):
    #paiming,status = get_chanye_status(city_name,chanye)
    sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `城市` like '%{city_name}%' and (`产业` like '%{chanye}%' or `产业节点` like '%{chanye}%');"
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
    paiming = chanye_rank[chanye]
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
    print(data)
