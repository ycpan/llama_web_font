from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd

conf = {
    "user": "default",
    "password": "",
    "server_host": "10.0.0.25",
    "port": "8123",
    "db": "algorithm"
}

def get_qiye_shuliang(city,chanye):
    sql = f"select count(`企业名称`) from `企业数据` where `城市` like '%{烟台}%' and `产业` like '%{chanye}%';"
    data = get_ck_data(sql)
    return data
def get_qiye_shuliang(city,chanye):
    sql = f"select `地区`,count(`企业名称`) as `企业数量` from `企业数据` where `城市` like '%{city}%' and `产业` like '%{chanye}%' and `地区` <> '' group by `地区` order by `企业数量` desc;"
    data = get_ck_data(sql)
    return data
def get_qiyeguimo(city,chanye):
    f = f'{city}{chanye}的数量为{count}个'
    
def get_ck_data(sql):
    connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
    engine = create_engine(connection, pool_size=100, pool_recycle=3600, pool_timeout=20)
    
    sql = text('SHOW TABLES')
    
    session = make_session(engine)
    data = []
    #cursor = session.execute(sql)
    try:
        fields = cursor._metadata.keys
        df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
        print(df)
        data = df
    finally:
        cursor.close()
        session.close()
    return data
if __name__ == "__main__":
    city = '烟台'
    data = get_qiyeguimo(city)
    print(data)

