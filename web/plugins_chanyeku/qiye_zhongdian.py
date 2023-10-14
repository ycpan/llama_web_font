#from zhishiku_mysql import find_by_sql
from .zhishiku_mysql import find_by_sql
def get_qiyezhongdian(city_name='烟台',chanye=''):
    res = []
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    data_sql = """select `企业名称`,`营收` from `企业数据` where  城市 like '%{}%'  and `产业` like "%{}%" order by `营收` desc limit 20;""".format(city_name,chanye)
    data = find_by_sql(data_sql)
    answer = f'{city_name}的重点企业有{int(0.5*len(data))}条,分别为'
    chanye = [da['企业名称'] for da in data[0:int(0.5*len(data))] if da['企业名称']]
    #for i in range(int(0.5*len(data))):
    #    if data[i][0]:
    #        answer += data[i][0]+'、'
    answer = f'{city_name}的重点企业共有{len(data)}家,分别为' + '、'.join(chanye)
    #print(answer)
    return answer
if __name__ == '__main__':
    data  = get_qiyezhongdian(chanye='新材料')
    print(data)
