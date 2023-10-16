#from chanye_plugins.zhishiku_mysql import find_by_sql
from .zhishiku_mysql import find_by_sql
#from zhishiku_mysql import find_by_sql
#def get_chanyechengshi(city_name='烟台',chanye=''):
def get_chanyechengshi(chanye=''):
    res = []
    """
    新能源产业比较强的城市有哪些？
    """
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    data_sql = f"""select `城市`,count(`城市`) as 数量 from `企业数据` where `产品` like "%{chanye}%" or `产业` like "%{chanye}%" or 产业节点 like "%{chanye}%" group by `城市` order by `数量` desc  limit 100;"""
    #import ipdb
    #ipdb.set_trace()
    data = find_by_sql(data_sql)
    chengshi = [da['城市'] for da in data[0:int(0.1*len(data))] if da['城市']]
    #chengshi = [da['城市'] for da in data if da['城市']]
    #answer = f'{chanye}比较强的城市有'
    answer = ''
    if chengshi:
        answer = answer + '、'.join(chengshi)
    #print(answer)
    return answer

def get_quanguopaiming(chanye='',top=10):
    res = []
    """
    新能源产业比较强的城市有哪些？
    """
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    data_sql = f"""select `城市`,count(`城市`) as 数量 from `企业数据` where `产品` like "%{chanye}%" or `产业` like "%{chanye}%" or 产业节点 like "%{chanye}%" group by `城市` order by `数量` desc  limit {top};"""
    #import ipdb
    #ipdb.set_trace()
    data = find_by_sql(data_sql)
    #chengshi = [da['城市'] for da in data[0:int(0.1*len(data))] if da['城市']]
    chengshi = [da['城市'] for da in data if da['城市']]
    #answer = f'{chanye}比较强的城市有'
    answer = ''
    if chengshi:
        answer = answer + '、'.join(chengshi)
    #print(answer)
    return answer
if __name__ == '__main__':
    data = get_chanyechengshi('新能源')
    print(data)
