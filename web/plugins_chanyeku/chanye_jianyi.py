import pickle
import requests
import json
import pandas as pd
import random
import os 
from .location_mapper import get_code_from_str
#from location_mapper import get_code_from_str
def get_chanye2code():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    index  = dir_path.find('/web')
    web_path = dir_path[0:index + 5]
    csv_path = os.path.join(web_path,'plugins_chanyeku/chanye2code.csv')
    data = pd.read_csv(csv_path)
    return data
payload = ""
headers = {}
#f = open('/devdata/home/user/panyongcan/Project/llama_web_font/web/plugins_chanyeku/industry2code.pkl','rb')
#industry2code = pickle.load(f)
#industrycode = {}
#for key in industry2code:
#    for sub_key in industry2code[key]:
#        industrycode[sub_key]=industry2code[key][sub_key]
#        industrycode[sub_key.replace('产业','')]=industry2code[key][sub_key]
data = get_chanye2code()
industrycode = {}
for _,da in data.iterrows():
    code = da['node']
    if len(code) != 3:
        continue
    chanye = da['name']
    industrycode[chanye]=code
    #chanye = chanye.replace('产业','')
    #industrycode[chanye]=code

def get_chanye_status(city_name,chanye=''):
    #url = "https://devdly.incostar.com/api/special/industrial/end/getIndustrialStatus?nodeCode=H03&areaCode=110100"
    if chanye and chanye in industrycode:
        node_code=industrycode[chanye]
    else:
        node_code = ''
    area_code = get_code_from_str(city_name)
    area_code=area_code.replace('0000','0100')
    url = f"https://devdly.incostar.com/api/special/industrial/end/getIndustrialStatus?nodeCode={node_code}&areaCode={area_code}"
    response = requests.request("GET", url, headers=headers, data=payload)
    data = json.loads(response.text)
    print(data)
    if data['code'] == 500:
        paiming = random.randint(60,100)
        status = '非优势产业'
        return paiming,status
    paiming = data['data']['rank']
    status = '优势产业' if data['data']['status'] == '1' else '非优势产业'
    return paiming,status
def get_chanyejianyi(city_name,chanye=''):
    youshi = {}
    lieshi = {}
    for cy in industrycode:
        paiming,status = get_chanye_status(city_name,cy)
        if not paiming:
            continue
        if paiming > 50:
            continue
        if status == '优势产业':
            youshi[cy]=paiming
        else:
            lieshi[cy]=paiming
    youshi_sorted = sorted(youshi.items(),key=lambda k:k[1])
    youshi_cy = [k for k,v in youshi_sorted]
    lieshi_sorted = sorted(lieshi.items(),key=lambda k:k[1])
    lieshi_cy = [k for k,v in lieshi_sorted]
    youshi_paiming = ['{}在全国的排名是第{}位'.format(cy,youshi[cy]) for cy in youshi_cy]
    lieshi_paiming = ['{}在全国的排名是第{}位'.format(cy,lieshi[cy]) for cy in lieshi_cy]
    all_chanye = '、'.join(youshi_cy+lieshi_cy)
    cy_paiming_str = '\n'.join(youshi_paiming + lieshi_paiming)
    #answer = f"""{city_name}当前存在的主要产业共有{len(youshi_cy) + len(lieshi_cy)}条，分别是{all_chanye}。
    #其中，从产业链企业规模看、从产业链重点企业规模看、从产业链发展趋势看：
    #{cy_paiming_str}
    #其中，{'、'.join(youshi_cy)}属于优势产业链，{'、'.join(lieshi_cy)}属于劣势产业链。 "
    #"""
    longtou = []
    zhongdian = []
    tisheng = []
    for cy,paiming in youshi_sorted:
        if paiming <= 10:
            longtou.append(cy)
        elif paiming <= 20:
            zhongdian.append(cy)
        elif paiming <=40:
            tisheng.append(cy)
    longtou_str="、".join(longtou)
    zhongdian_str="、".join(zhongdian)
    tisheng_str="、".join(tisheng)
    answer = ""
    if longtou:
        longtou_jianyi = f"""
对{longtou_str}等产业重点推进以下工作：保持当前产业优势，发挥产业带头作用；支持{longtou[0]}等产业开拓海外市场，强化{longtou[0]}的优势作用；扩大{longtou_str}在国内的影响力，吸引更多上下游企业。"""
        answer += longtou_jianyi + '\n'
    if zhongdian:
        zhongdian_jianyi = f"""对{zhongdian_str}等产业重点推进以下工作：加大本区域重点企业培育工作；扩展相关产业在全国的影响力；扩大相关产业企业的引入。"""
        answer += zhongdian_jianyi + '\n'
    if tisheng:
        tisheng_jianyi = f"""对于{tisheng_str}等进行升链，首先推动产业链升级。对于不具备升级条件的，推动产业链转型、或者转型+升级。 """
        answer += tisheng_jianyi
    if not answer:
        answer = f'由于我的数据不是太完善，当前产业或者地区排名，我也不知道哇'
    return answer
if __name__ == '__main__':
    city_name='烟台'
    chanye='新能源'
    #data = get_chanyejianyi(city_name='烟台',chanye='新能源')
    data = get_chanyejianyi(city_name='烟台')
    #data = get_chanyepaiming(city_name='北京',chanye='新能源')
    print(data)
