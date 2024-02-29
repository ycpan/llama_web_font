from tqdm import tqdm
import traceback
import pymysql
import hashlib
import pandas as pd
#hl.update(str.encode(encoding='utf-8'))
def get_md5(str_s):
    str_s = str_s.strip('. 。？')
    hl = hashlib.md5()
    hl.update(str_s.encode(encoding='utf-8'))
    res = hl.hexdigest()
    return res
class batchSql:
    def __init__(self,host,port,user,passwd,db_name,charset='utf8mb4'):
        param_mysql_bj = {}
        #param_mysql_bj['host'] = '10.0.0.11'
        #param_mysql_bj['port'] = 3306
        #param_mysql_bj['user'] = 'root'
        #param_mysql_bj['db'] = 'algorithm'
        param_mysql_bj['host'] = host
        param_mysql_bj['port'] = port
        param_mysql_bj['user'] = user
        param_mysql_bj['passwd'] = passwd
        param_mysql_bj['db'] = db_name
        param_mysql_bj['charset'] = charset
        param_mysql_bj['cursorclass'] = pymysql.cursors.DictCursor
        self.param_mysql_bj = param_mysql_bj
        self.connection= pymysql.connect(**self.param_mysql_bj)
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
        connection= pymysql.connect(**self.param_mysql_bj)
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
        res = []
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query_sql)
                for item in tqdm(cursor.fetchall()):
                    #name = item.get('companyName')
                    res.append(item)
            if not res:
                raise ValueError('connection may be lost')
            return res
        except:
            self.connection= pymysql.connect(**self.param_mysql_bj)
            with self.connection.cursor() as cursor:
                cursor.execute(query_sql)
                for item in tqdm(cursor.fetchall()):
                    #name = item.get('companyName')
                    res.append(item)
            return res


    def query_big_data(self,all_count):
        connection= pymysql.connect(**self.param_mysql_bj)
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
#db_name = 'algorithm'
host = '10.0.0.14'
port = 3306
user = 'root'
passwd = 'Incostar@2021'
db_name = 'algorithm_app'
my_sql = batchSql(host,port,user,passwd,db_name)
#batch_sql =  "INSERT INTO llm_web(code,question,query_web,content,answer,evaluation) VALUES(%s,%s,%s,%s,%s,%s)"
#
#| time           | datetime     | YES  |     | NULL    | on update CURRENT_TIMESTAMP |
#| question       | tinytext     | YES  |     | NULL    |                             |
#| agent_response | text         | YES  |     | NULL    |                             |
#| context_data   | longtext     | YES  |     | NULL    |                             |
#| final_response | text         | YES  |     | NULL    |                             |
#| feedback       | varchar(255) | YES  |     | NULL    |                             |
#| source         | varchar(255) | YES  |     | NULL    |                             |
#| userid         | int(11)      | YES  |     | NULL    |                             |
#| username
#
#batch_sql =  "INSERT INTO llm_log(question,agent_response,query_web,context_data,final_response,feedback,source) VALUES(%s,%s,%s,%s,%s,%s,%s)"
batch_sql =  "INSERT INTO llm_log(question,agent_response,context_data,final_response,is_normal) VALUES(%s,%s,%s,%s,%s)"
my_sql.set_batch_sql(batch_sql)
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
    data = my_sql.query_data(query_sql)
    if len(data) > 0:
        return data
    return data
def save(question,query_web,content,answer,evaluation):
    #import ipdb
    #ipdb.set_trace()
    code = get_md5(question)
    #value = (code,question,query_web,content,answer,evaluation)
    value = (question,query_web,content,answer,evaluation)
    my_sql.add_value(value)
    my_sql.batch_execute()
if __name__ == '__main__':

    question="漯河市产业结构"
    code = get_md5(question)
    print(code)
    query_web="漯河市产业结构"
    content="""
    在一系列创新扶持政策助力下，去年，该区利通科技成为全国首批81家在北交所上市的企业之一，实现了漯河多年来未有新主板上市企业的突破；卫龙公司发展成漯河市第二家营收超100亿元的本土企业；通过“小升规”培育工程，35家小企业得以提档升级。

    以创新布局未来产业。围绕激发市场主体创新活力，积极发展高新技术产业、未来产业，漯河经开区持续加大科技研发投入力度，财政科技支出占一般公共预算支出比例保持在7%左右。去年，该区社会研发投入近11亿元，同比增长43.6%，基本实现规上工业研发活动全覆盖。目前，该区内国家级高新技术企业达31家，有50家创新型企业入库国家科技型中小企业，占到了漯河市的半壁江山。"""
    answer=''
    evaluation=''

    #value = (self.transform_value(line['question']),self.transform_value(line['query_web']),self.transform_value(line['content']),self.transform_value(line['answer']),self.transform_value(line['evaluation']))
    value = (code,question,query_web,content,answer,evaluation)
    #my_sql.add_value(value)
    #my_sql.batch_execute()
    query_sql = 'select * from llm_web'
    data = my_sql.query_data(query_sql)
    print(data)
    
