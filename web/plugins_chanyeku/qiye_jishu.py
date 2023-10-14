import re
#from zhishiku_mysql import find_by_sql
from .zhishiku_mysql import find_by_sql
#from zhishiku_mysql import find_by_sql
def get_qiyejishu(city_name='烟台',chanye=''):
    res = []
    """
    烟台新能源产业的主要技术有哪些？
    """
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    data_sql = """select `产业节点` from `企业数据` where  城市 like '%{}%'  and `产业` like "%{}%" and `产业节点` is not null order by `营收` desc limit 50;""".format(city_name,chanye)
    if not chanye:
        data_sql = """select `产业节点` from `企业数据` where  城市 like '%{}%'  and `产业节点` is not null order by `营收` desc limit 50;""".format(city_name,chanye)
    data = find_by_sql(data_sql)
    answer = f'{city_name}的{chanye}的技术有'
    jishu = [da['产业节点'] for da in data[0:int(0.5*len(data))] if da['产业节点']]
    #for i in range(int(0.5*len(data))):
    #    if data[i][0]:
    #        answer += data[i][0]+'、'
    res_jishu = {}
    for cp in jishu:
        cp_li = re.split('、|;|；',cp)
        for sub_cp in cp_li:
            if len(sub_cp) > 8:
                continue
            if sub_cp.strip() == '-':
                continue
            if sub_cp not in res_jishu:
                res_jishu[sub_cp] = 0
            res_jishu[sub_cp] += 1
    sorted_jishu = sorted(res_jishu.items(),key=lambda k:k[1],reverse=True)

    jishu = []
    for cp,count in sorted_jishu:
        if len(jishu) > 5:
            break
        if cp.strip():
            jishu.append(cp)
    answer = answer + ','.join(jishu)
    #print(answer)
    return answer
if __name__ == '__main__':
    data = get_qiyejishu('烟台','新能源')
    #data = get_qiyejishu('烟台')
    print(data)
