import re
import pandas as pd
from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine
#from zhishiku_mysql import find_by_sql
#from chanye_paiming import get_chanye_status
#from .zhishiku_mysql import find_by_sql
#from .chanye_paiming import get_chanye_status
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
def process_data(city_name,data):
    res = []

    chanye_rank = {}
    for _,da in data.iterrows():
        cy = da['产业']
        paiming = da['排名']
        if cy not in chanye_rank:
            chanye_rank[cy] = []
        chanye_rank[cy].append(paiming)
    for cy in chanye_rank:
        chanye_rank[cy]=int(sum(chanye_rank[cy])/len(chanye_rank[cy]))
    youshi = {}
    lieshi = {}
    mid = {}
    sorted_chanye_rank = sorted(chanye_rank.items(),key=lambda k:k[1])
    for cy,paiming in sorted_chanye_rank:
        #paiming,status = get_chanye_status(city_name,cy)
        #paiming = chanye_rank[cy]
        if cy == '全部产业链':
            continue
        if paiming <= 30:
            youshi[cy]=paiming
            continue
        if chanye_rank[cy] > 60:
            lieshi[cy]=paiming
            continue
        mid[cy]=paiming
    #li = sorted(res)
    #li = sorted(res.items(),key=lambda k:k[1])
    youshi_li = [k for k in youshi]
    mid_li = [k for k in mid]
    lieshi_li = [k for k in lieshi]
    you_answer = []
    answer = ''
    if youshi_li:
        answer += '优势产业有{}'.format('、'.join(youshi_li))
        for cy in youshi_li:
            #ya = f"{cy}全国排名第{chanye_rank['优势'][cy]}位"
            ya = f"{cy}全国排名第{chanye_rank[cy]}位"
            you_answer.append(ya)
        answer += '。其中:\n{}'.format('\n'.join(you_answer)) + '。'
    if mid_li:
        answer += '非优势产业有{}'.format('、'.join(mid_li))
        mid_answer = []
        for cy in mid_li:
            #ya = f"{cy}全国排名第{chanye_rank['优势'][cy]}位"
            ya = f"{cy}全国排名第{chanye_rank[cy]}位"
            mid_answer.append(ya)
        answer += '。其中:\n{}'.format('\n'.join(mid_answer)) + '。'
    if lieshi_li:
        if answer:
            answer += '\n\n劣势产业有{}'.format('、'.join(lieshi_li))
        else:
            answer = '劣势产业有{}'.format('、'.join(lieshi_li))
        lie_answer = []
        for cy in lieshi_li:
            #ya = f"{cy}全国排名第{chanye_rank['劣势'][cy]}位"
            ya = f"{cy}全国排名第{chanye_rank[cy]}位"
            lie_answer.append(ya)
        answer += '。其中:\n{}'.format('\n'.join(lie_answer)) + '。'
    if not lieshi_li and not youshi_li:
        answer = ''
    return answer

def get_chanyezhenduan(city_name='烟台',chanye=''):
    """
    烟台重点产业有哪些？烟台产业发展状况如何?
    """
    res = []
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    #data_sql = "select 产业 from 企业数据 where 城市 like '%{}%'  and 营收 > 10 limit 2000;".format(city_name)
    sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `城市` like '%{city_name}%';"
    data = get_ck_data(sql)
    #data = find_by_sql(data_sql)
    answer = process_data(city_name,data)
    #answer = f'{city_name}的产业链共有{len(data)}条,其中重点产业有{int(0.5*len(data))}条,分别为'
    #chanye = [da['产业'] for da in data[0:int(0.5*len(data))] if da['产业']]
    #answer = f'{city_name}的产业链共有{len(data)}条,其中重点产业有{int(len(chanye))}条,分别为:' + ';'.join(chanye)
    return answer
if __name__ == '__main__':
    data = get_chanyezhenduan()
    print(data)
