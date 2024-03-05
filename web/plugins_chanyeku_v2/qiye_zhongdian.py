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
def get_qiyezhongdian(city_name='烟台',chanye=''):
    res = []
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    #data_sql = """select `企业名称`,`营收` from `企业数据` where  城市 like '%{}%'  and `产业` like "%{}%" order by `营收` desc limit 20;""".format(city_name,chanye)
    #data_sql = """select `企业名称`,`产业评分` from `Qi_Ye_output` where (`省份` like '%city_name%' or `城市` like '%city_name%' or `区域` like '%city_name%') and `产业节点` like '%chanye%' order by `产业评分` desc limit 100;""".format(city_name,chanye)
    data_sql = f"""select `企业名称`,`产业评分` from `Qi_Ye_output` where (`省份` like '%{city_name}%' or `城市` like '%{city_name}%' or `区域` like '%{city_name}%') and `产业节点` like '%{chanye}%' order by `产业评分` desc limit 100;"""
    if not city_name and chanye:
        data_sql = f"""select `企业名称`,`产业评分` from `Qi_Ye_output` where `产业节点` like '%{chanye}%' order by `产业评分` desc limit 100;"""
    if  city_name and not  chanye:
        data_sql = f"""select `企业名称`,`产业评分` from `Qi_Ye_output` where `省份` like '%{city_name}%' or `城市` like '%{city_name}%' or `区域` like '%{city_name}%' order by `产业评分` desc limit 100;"""
    #data = find_by_sql(data_sql)
    data = get_ck_data(data_sql)
    #import ipdb
    #ipdb.set_trace()
    #answer = f'{city_name}的{chanye}重点企业有{int(0.5*len(data))}条,分别为'
    qiyes = []
    for da in data:
        if len(qiyes) < 10 or da['产业评分'] > 70:
            qiyes.append(da['企业名称'])
    #chanye = [da['企业名称'] for da in data[0:int(0.5*len(data))] if da['企业名称']]
    #for i in range(int(0.5*len(data))):
    #    if data[i][0]:
    #        answer += data[i][0]+'、'
    answer = f'{city_name}的{chanye}重点企业共有{len(qiyes)}家,分别为' + '、'.join(qiyes) + '。'
    if not chanye:
        answer = ''
    #print(answer)
    return answer
if __name__ == '__main__':
    data  = get_qiyezhongdian(city_name='江苏',chanye='新能源')
    print(data)
