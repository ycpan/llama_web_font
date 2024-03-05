#from zhishiku_mysql import find_by_sql
#from .zhishiku_mysql import find_by_sql
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
        #df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
        #data = df
        data = [dict(zip(fields, item)) for item in cursor.fetchall()]
    finally:
        cursor.close()
        session.close()
    return data
def get_chanyequshi(city_name='烟台',chanye=''):
    res = []
    #data_sql = f"""select `企业名称`,`产业评分` from `Qi_Ye_output` where (`省份` like '%{city_name}%' or `城市` like '%{city_name}%' or `区域` like '%{city_name}%') and `产业节点` like '%{chanye}%' order by `产业评分` desc limit 100;"""
    data_sql = f"""select YEAR(`注册时间`) as founded_year,count(*) as count from `Qi_Ye_output` group by founded_year order by founded_year desc limit 10;"""
    if not city_name and chanye:
        #data_sql = f"""select `企业名称`,`产业评分` from `Qi_Ye_output` where `产业节点` like '%{chanye}%' order by `产业评分` desc limit 100;"""
        data_sql = f"""select YEAR(`注册时间`) as founded_year,count(*) as count from `Qi_Ye_output` group by founded_year order by founded_year desc limit 10;"""
    if  city_name and not  chanye:
        #data_sql = f"""select `企业名称`,`产业评分` from `Qi_Ye_output` where `省份` like '%{city_name}%' or `城市` like '%{city_name}%' or `区域` like '%{city_name}%' order by `产业评分` desc limit 100;"""
        data_sql = f"""select YEAR(`注册时间`) as founded_year,count(*) as count from `Qi_Ye_output` group by founded_year order by founded_year desc limit 10;"""
    #data = find_by_sql(data_sql)
    data = get_ck_data(data_sql)
    #import ipdb
    #ipdb.set_trace()
    #answer = f'{city_name}的{chanye}重点企业有{int(0.5*len(data))}条,分别为'
    if not data:
        return ''
    qiyes = []
    for da in data:
        year,count = da['founded_year'],da['count']
        qiyes.append(f'{city_name}的{chanye}企业在{year}新增{count}家')
    #chanye = [da['企业名称'] for da in data[0:int(0.5*len(data))] if da['企业名称']]
    #for i in range(int(0.5*len(data))):
    #    if data[i][0]:
    #        answer += data[i][0]+'、'
    #answer = f'{city_name}的{chanye}重点企业共有{len(qiyes)}家,分别为' + '、'.join(qiyes) + '。'
    answer = '；\n'.join(qiyes) + '。'
    if not chanye:
        answer = ''
    #print(answer)
    return answer
if __name__ == '__main__':
    data = get_chanyequshi(city_name='江苏',chanye='新能源')
    print(data)
