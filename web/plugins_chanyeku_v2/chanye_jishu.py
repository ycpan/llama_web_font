import re
from clickhouse_sqlalchemy import make_session
import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import create_engine
#from zhishiku_mysql import find_by_sql
#from .zhishiku_mysql import find_by_sql
#from zhishiku_mysql import find_by_sql
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
def get_chanyejishu(city_name='烟台',chanye=''):
    res = []
    """
    烟台新能源产业的主要技术有哪些？
    """
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    #data_sql = """select `产业节点` from `企业数据` where  城市 like '%{}%'  and `产业` like "%{}%" and `产业节点` is not null order by `营收` desc limit 50;""".format(city_name,chanye)
    data_sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `城市` like '%{city_name}%' and `产业` like '%{chanye}%';"
    if not chanye:
        #data_sql = """select `产业节点` from `企业数据` where  城市 like '%{}%'  and `产业节点` is not null order by `营收` desc limit 50;""".format(city_name,chanye)
        data_sql = f"select `城市`,`产业`,`产业节点`,`排名` from `产业诊断` where `城市` like '%{city_name}%';"
    #data = find_by_sql(data_sql)
    data = get_ck_data(data_sql)
    cy_nd_rank = {}
    for _,da in data.iterrows():
        cy_nd = da['产业节点']
        paiming = da['排名']
        if chanye in cy_nd or cy_nd in chanye:
            continue
        if cy_nd not in cy_nd_rank:
            cy_nd_rank[cy_nd] = paiming
    sorted_cy_nd_rank = sorted(cy_nd_rank.items(),key=lambda k:k[1])
    #jishu = [da['产业节点'] for da in data[0:int(0.5*len(data))] if da['产业节点']]
    jishu = []
    for nd,paiming in sorted_cy_nd_rank:

        nd = nd.replace('产业','') 
        nd = nd + '技术'
        if len(jishu) < 2 and paiming < 50:
            if nd not in jishu:
                jishu.append(nd)
        if paiming < 10:
            if nd not in jishu:
                jishu.append(nd)

    answer = f'{city_name}的{chanye}的技术有'
    #for i in range(int(0.5*len(data))):
    #    if data[i][0]:
    #        answer += data[i][0]+'、'
    #res_jishu = {}
    #for cp in jishu:
    #    cp_li = re.split('、|;|；',cp)
    #    for sub_cp in cp_li:
    #        if len(sub_cp) > 8:
    #            continue
    #        if sub_cp.strip() == '-':
    #            continue
    #        if sub_cp not in res_jishu:
    #            res_jishu[sub_cp] = 0
    #        res_jishu[sub_cp] += 1
    #sorted_jishu = sorted(res_jishu.items(),key=lambda k:k[1],reverse=True)

    #jishu = []
    #for cp,count in sorted_jishu:
    #    if len(jishu) > 5:
    #        break
    #    if cp.strip():
    #        jishu.append(cp)
    answer = answer + ','.join(jishu)
    if not jishu:
        answer = ''
    #print(answer)
    return answer
if __name__ == '__main__':
    data = get_chanyejishu('烟台','新能源')
    #data = get_chanyejishu('烟台')
    print(data)
