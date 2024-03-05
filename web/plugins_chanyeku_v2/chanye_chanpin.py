import re
#from zhishiku_mysql import find_by_sql
from .zhishiku_mysql import find_by_sql
def get_chanyechanpin(city_name='烟台',chanye=''):
    """
    烟台市新能源汽车的主要产品有哪些？
    """
    res = []
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    data_sql = """select `产品` from `企业数据` where  城市 like '%{}%'  and `产业` like "%{}%" and `产品` is not null order by `营收` desc limit 50;""".format(city_name,chanye)
    if not chanye:
        data_sql = """select `产品` from `企业数据` where  城市 like '%{}%' and `产品` is not null order by `营收` desc limit 50;""".format(city_name,chanye)
    #import ipdb
    #ipdb.set_trace()
    data = find_by_sql(data_sql)
    answer = f'{city_name}的{chanye}主要产品有'
    #import ipdb
    #ipdb.set_trace()
    #chanpin = [da['产品'] for da in data[0:int(0.5*len(data))] if da['产品']]
    chanpin = [da['产品'] for da in data if da['产品']]
    res_chanpin = {}
    for cp in chanpin:
        cp_li = re.split('、|;|；|。|，|,',cp)
        for sub_cp in cp_li:
            if len(sub_cp) > 8 or len(sub_cp) < 2:
                continue
            if sub_cp not in res_chanpin:
                res_chanpin[sub_cp] = 0
            res_chanpin[sub_cp] += 1
    sorted_chanpin = sorted(res_chanpin.items(),key=lambda k:k[1],reverse=True)

    chanpin = []
    for cp,count in sorted_chanpin:
        if len(chanpin) > 5:
            break
        if cp.strip():
            chanpin.append(cp)
    #for i in range(int(0.5*len(data))):
    #    if data[i][0]:
    #        answer += data[i][0]+'、'
    answer = answer + ','.join(chanpin)
    #print(answer)
    if not chanpin:
        answer = ''
    return answer
if __name__ == '__main__':
    #res = get_chanyechanpin('烟台','新能源')
    #res = get_chanyechanpin('烟台','')
    res = get_chanyechanpin('北京','新能源')
    #res = get_chanyechanpin('万华','化工')
    print(res)
