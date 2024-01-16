from tqdm import tqdm
import traceback
import hashlib
import pandas as pd
from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine

#hl.update(str.encode(encoding='utf-8'))
def get_md5(str_s):
    str_s = str_s.strip('. 。？')
    hl = hashlib.md5()
    hl.update(str_s.encode(encoding='utf-8'))
    res = hl.hexdigest()
    return res
class batchSql:
    def __init__(self,host,port,user,passwd,db_name,charset='utf8mb4'):
        param_ck_bj = {}
        #param_ck_bj['host'] = '10.0.0.11'
        #param_ck_bj['port'] = 3306
        #param_ck_bj['user'] = 'root'
        #param_ck_bj['db'] = 'algorithm'
        #param_ck_bj['host'] = host
        #param_ck_bj['port'] = port
        #param_ck_bj['user'] = user
        #param_ck_bj['pw'] = passwd
        #param_ck_bj['database'] = db_name
        #param_ck_bj['charset'] = charset
        #param_ck_bj['cursorclass'] = pyck.cursors.DictCursor
        self.param_ck_bj = param_ck_bj
        #self.connection= pyck.connect(**self.param_ck_bj)
        import ipdb
        ipdb.set_trace()
        #self.connection = connect(**self.param_ck_bj)
        #host = '10.0.0.25'
        #port = 8123
        #user = 'default'
        #pw = ''
        #database = 'algorithm'
        conf = {
            "user": "default",
            "password": "",
            "server_host": "10.0.0.25",
            "port": "8123",
            "db": "algorithm"
        }
        #self.connection = connect(f'clickhouse://{user}:{pw}@{host}:9000/{database}')
        self.connection_url = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
        self.connection = create_engine(self.connection_url, pool_size=100, pool_recycle=3600, pool_timeout=20)
        self.batch_insert_sql =  None
        self.values = []
    def set_batch_sql(self,sql):
        """
        batch_insert_sql =  "INSERT INTO company(companyName,code,website,scope,descrb,companyLevel,dataSource,yewuTags) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
        or
        batch_repalce_into =   "REPLACE INTO company(companyName,code,website,scope,descrb,companyLevel,dataSource,yewuTags) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
        or 
        batch_update_sql = "update news set tags=(%s) where docid=(%s)"
        """
        if not sql:
            self.batch_insert_sql = "INSERT INTO company(companyName,code,website,scope,descrb,companyLevel,dataSource,yewuTags) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"
        else:
            self.batch_insert_sql = sql
    def add_value(self,tuple_values):
        self.values.append(tuple_values)
    def clean_value(self):
        self.values = []
    @property
    def batch_length(self):
        return len(self.values)
    def batch_execute(self):
        # 使用 cursor() 方法创建一个游标对象 cursor
        connection= pyck.connect(**self.param_ck_bj)
        cursor = connection.cursor()
        try:
           # 执行sql语句
           cursor.executemany(self.batch_insert_sql,self.values)
           # 提交到数据库执行
           connection.commit()
        except Exception as e:
           # 如果发生错误则回滚
           print(self.values)
           tb = traceback.format_exc()
           print(tb)
           connection.rollback()
           self.clean_value()
        connection.close()
        self.values = []
        #self.batch_insert_sql = None
    def query_data(self,query_sql):
        query_sql = text(query_sql)
        res = []
        import ipdb
        ipdb.set_trace()
        try:
            with make_session(self.connection) as session:
                cursor = session.execute(query_sql)
                fields = cursor._metadata.keys
                for item in tqdm(cursor.fetchall()):
                    #name = item.get('companyName')
                    res.append(dict(zip(fields,item)))
            if not res:
                raise ValueError('connection may be lost')
            return res
        except:
            self.connection = create_engine(self.connection_url, pool_size=100, pool_recycle=3600, pool_timeout=20)
            with make_session(self.connection) as session:
                cursor = session.execute(query_sql)
                fields = cursor._metadata.keys
                for item in tqdm(cursor.fetchall()):
                    #name = item.get('companyName')
                    #res.append(item)
                    res.append(dict(zip(fields,item)))
            return res


    def query_big_data(self,all_count):
        connection= connect(**self.param_ck_bj)
        with connection.cursor() as cursor:
            cnt = 0
            while cnt < all_count:
                sql = "select content,title,docid from news limit 500 offset {}".format(cnt)
                cnt += 500
                print('process {}/{}'.format(cnt,259190))
                result = []
                cursor.execute(sql)
                for item in cursor.fetchall():
                    content = item['content']
                    title = item['title']
                    input_text = title + content
                    tags = tag_news(input_text)
                    docid = item['docid']
                    result.append((','.join(tags),docid))

#host = '10.0.0.11'
#port = 3306
#user = 'root'
#passwd = 'Incostar@2021'
#db_name = 'algorithm_app'
##db_name = 'algorithm'
host = '10.0.0.25'
port = 8123
user = 'default'
passwd = ''
db_name = 'algorithm'
#db_name = 'algorithm'
my_sql = batchSql(host,port,user,passwd,db_name)
#batch_sql =  "select count(distinct `企业名称`) as `企业数量` from `企业数据` where `企业类型` like '%专精特新%' and `城市` like '%北京%';"
batch_sql =  "select * from `企业数据` where `企业类型` like '%专精特新%' and `城市` like '%北京%';"
#my_sql.set_batch_sql(batch_sql)
def find(s,step=0):
    code = get_md5(s)
    query_sql = 'select content from llm_web where code=\'{}\' limit 1;'.format(code)
    data = my_sql.query_data(query_sql)
    #print(data)
    #return data
    if len(data) > 0:
        return data[0]['content']
    return data
def find_by_sql(query_sql,step=0):
    import ipdb
    ipdb.set_trace()
    data = my_sql.query_data(query_sql)
    if len(data) > 0:
        return data
    return data
def save(question,query_web,content,answer,evaluation):
    code = get_md5(question)
    value = (code,question,query_web,content,answer,evaluation)
    my_sql.add_value(value)
    my_sql.batch_execute()
if __name__ == '__main__':

    #question="漯河市产业结构"
    #code = get_md5(question)
    #print(code)
    #query_web="漯河市产业结构"
    #content="""
    #在一系列创新扶持政策助力下，去年，该区利通科技成为全国首批81家在北交所上市的企业之一，实现了漯河多年来未有新主板上市企业的突破；卫龙公司发展成漯河市第二家营收超100亿元的本土企业；通过“小升规”培育工程，35家小企业得以提档升级。

    #以创新布局未来产业。围绕激发市场主体创新活力，积极发展高新技术产业、未来产业，漯河经开区持续加大科技研发投入力度，财政科技支出占一般公共预算支出比例保持在7%左右。去年，该区社会研发投入近11亿元，同比增长43.6%，基本实现规上工业研发活动全覆盖。目前，该区内国家级高新技术企业达31家，有50家创新型企业入库国家科技型中小企业，占到了漯河市的半壁江山。"""
    #answer=''
    #evaluation=''

    ##value = (self.transform_value(line['question']),self.transform_value(line['query_web']),self.transform_value(line['content']),self.transform_value(line['answer']),self.transform_value(line['evaluation']))
    #value = (code,question,query_web,content,answer,evaluation)
    ##my_sql.add_value(value)
    ##my_sql.batch_execute()
    #query_sql = 'select * from llm_web'
    #data = my_sql.query_data(query_sql)
    #print(data)
    data = find_by_sql(batch_sql)
    print(data)
    
