from tqdm import tqdm
import traceback
import pymysql
import hashlib
import pandas as pd
from neo4j import GraphDatabase

class MyNeo4j:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # 定义查询函数
    def query_data(self,cypher_sql):
        query = """
        MATCH (e:Employee {id: $id})<-[:SUPERVISES]-(subordinate)
        RETURN subordinate
        """
        with self.driver.session() as session:
            #result = session.run(query, id=employee_id)
            result = session.run(cypher_sql)
            #import ipdb
            #ipdb.set_trace()
            data = result.data()
            return data
myneo4j = MyNeo4j("bolt://10.0.0.14:17687", "neo4j", "123456789")
def find(s,step=0):
    code = get_md5(s)
    query_sql = 'select content from llm_web where code=\'{}\' limit 1;'.format(code)
    data = myneo4j.query_data(query_sql)
    #print(data)
    #return data
    #if len(data) > 0:
    #    return data[0]['content']
    return data
def find_by_sql(query_sql,step=0):
    data = myneo4j.query_data(query_sql)
    if len(data) > 0:
        return data
    return data
def save(question,query_web,content,answer,evaluation):
    code = get_md5(question)
    value = (code,question,query_web,content,answer,evaluation)
    myneo4j.add_value(value)
    myneo4j.batch_execute()
if __name__ == "__main__":
    #cy_sql = "match (current {name:'内存条'}) -[:`下级`]->(children) return current.name,children.name;"
    #cy_sql = "match (current {name:'内存条'}) -[:`下级`*1..3]->(children) return current.name,children.name;"
    cy_sql = "match (current {name:'内存条'}) -[:`下级`]->(level1)-[:`下级`]->(level2)-[:`下级`]->(level3) return current.name,level1.name,level2.name,level3.name;"
    #res = myneo4j.query_data(cy_sql)
    #myneo4j.close()
    res = find_by_sql(cy_sql)
    print(res)
    
