#from chanye_plugins.zhishiku_mysql import find_by_sql
#from .zhishiku_mysql import find_by_sql
#from zhishiku_mysql import find_by_sql
#def get_chanyechengshi(city_name='烟台',chanye=''):
import pandas as pd
from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine
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
#def get_chanyechengshi(chanye=''):
def get_quanguopaiming(chanye='',top=10):
    res = []
    if isinstance(top,str):
        top = int(top)
    """
    新能源产业比较强的城市有哪些？
    """
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    #data_sql = f"""select `城市`,count(`城市`) as 数量 from `企业数据` where `产品` like "%{chanye}%" or `产业` like "%{chanye}%" or 产业节点 like "%{chanye}%" group by `城市` order by `数量` desc  limit 100;"""
    #import ipdb
    #ipdb.set_trace()
    #data = find_by_sql(data_sql)
    sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `产业` like '%{chanye}%' or `产业节点` like '%{chanye}%';"
    data = get_ck_data(sql)
    chanye_rank = {}
    for _,da in data.iterrows():
        city = da['城市']
        paiming = da['排名']
        if city not in chanye_rank:
            chanye_rank[city] = []
        chanye_rank[city].append(paiming)
    for city in chanye_rank:
        chanye_rank[city]=int(sum(chanye_rank[city])/len(chanye_rank[city]))
    sorted_chanye_rank = sorted(chanye_rank.items(),key = lambda k:k[1])
    chengshi = []
    for city,paiming in sorted_chanye_rank:
        chengshi.append(city)
        if len(chengshi) >= top:
            break
    #chengshi = [da['城市'] for da in data[0:int(0.1*len(data))] if da['城市']]
    #chengshi = [da['城市'] for da in data if da['城市']]
    prefix_answer = f'{chanye}比较强的城市有'
    answer = ''
    if chengshi:
        answer = prefix_answer + '、'.join(chengshi)
    #print(answer)
    return answer

def get_chanyechengshi(chanye=''):
    res = []
    """
    新能源产业比较强的城市有哪些？
    """
    data = get_quanguopaiming(chanye)
    return data
if __name__ == '__main__':
    #data = get_chanyechengshi('新能源')
    data = get_quanguopaiming('新能源',20)
    print(data)
