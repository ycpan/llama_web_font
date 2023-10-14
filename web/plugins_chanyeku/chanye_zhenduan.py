import re
#from zhishiku_mysql import find_by_sql
#from chanye_paiming import get_chanye_status
from .zhishiku_mysql import find_by_sql
from .chanye_paiming import get_chanye_status
def process_data(city_name,data):
    res = []
    for da in data:
        chanye = da['产业']
        if not chanye:
            continue
        li = re.split(',|，|、',chanye)
        res.extend(li)
    chanye = list(set(res))
    #res = {'优势':{},'劣势':{}}
    #for cy in chanye:
    #    paiming,status = get_chanye_status(city_name,cy)
    #    if status == '优势产业':
    #        res['优势'][cy]=paiming
    #    else:
    #        res['劣势'][cy]=paiming
    #youshi_li = sorted(res['优势'],reverse=True)
    #lieshi_li = sorted(res['劣势'],reverse=True)
    res = {}
    for cy in chanye:
        paiming,status = get_chanye_status(city_name,cy)
        res[cy]=paiming
    #li = sorted(res)
    li = sorted(res.items(),key=lambda k:k[1])
    youshi_li = [x for x,y in li[0:int(0.3 * len(li))]]
    lieshi_li = [x for x,y in li[int(0.3 * len(li)):]]
    you_answer = []
    answer = ''
    if youshi_li:
        answer += '优势产业有{}'.format('、'.join(youshi_li))
        for cy in youshi_li:
            #ya = f"{cy}全国排名第{res['优势'][cy]}位"
            ya = f"{cy}全国排名第{res[cy]}位"
            you_answer.append(ya)
        answer += '。其中:\n{}'.format('\n'.join(you_answer)) + '。'
    if lieshi_li:
        if answer:
            answer += '\n\n劣势产业有{}'.format('、'.join(lieshi_li))
        else:
            answer = '劣势产业有{}'.format('、'.join(lieshi_li))
        lie_answer = []
        for cy in lieshi_li:
            #ya = f"{cy}全国排名第{res['劣势'][cy]}位"
            ya = f"{cy}全国排名第{res[cy]}位"
            lie_answer.append(ya)
        answer += '。其中:\n{}'.format('\n'.join(lie_answer)) + '。'
    return answer

def get_chanyezhenduan(city_name='烟台',chanye=''):
    """
    烟台重点产业有哪些？烟台产业发展状况如何?
    """
    res = []
    #city_name = questions_list[0].split('市')[0] + '市'
    #city_name = '烟台'
    #print(city_name)
    data_sql = "select 产业 from 企业数据 where 城市 like '%{}%'  and 营收 > 10 limit 2000;".format(city_name)
    data = find_by_sql(data_sql)
    answer = process_data(city_name,data)
    #answer = f'{city_name}的产业链共有{len(data)}条,其中重点产业有{int(0.5*len(data))}条,分别为'
    #chanye = [da['产业'] for da in data[0:int(0.5*len(data))] if da['产业']]
    #answer = f'{city_name}的产业链共有{len(data)}条,其中重点产业有{int(len(chanye))}条,分别为:' + ';'.join(chanye)
    return answer
if __name__ == '__main__':
    data = get_chanyezhenduan()
    print(data)
