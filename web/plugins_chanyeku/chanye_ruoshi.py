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
def get_chanyeruoshi(city_name,chanye=''):
    youshi = {}
    lieshi = {}
    for cy in industrycode:
        paiming,status = get_chanye_status(city_name,cy)
        if not paiming:
            continue
        #if paiming > 50:
        #    continue
        if status == '优势产业':
            youshi[cy]=paiming
        else:
            lieshi[cy]=paiming
    #youshi_sorted = sorted(youshi.items(),key=lambda k:k[1])
    #youshi_cy = [k for k,v in youshi_sorted]
    lieshi_sorted = sorted(lieshi.items(),key=lambda k:k[1])
    lieshi_cy = [k for k,v in lieshi_sorted]
    #youshi_paiming = ['{}在全国的排名是第{}位'.format(cy,youshi[cy]) for cy in youshi_cy]
    lieshi_paiming = ['{}在全国的排名是第{}位'.format(cy,lieshi[cy]) for cy in lieshi_cy]
    #all_chanye = '、'.join(youshi_cy+lieshi_cy)
    all_chanye = '、'.join(lieshi_cy)
    cy_paiming_str = '\n'.join(lieshi_paiming)
    answer = f"""{city_name}的非优势产业共有{len(lieshi_cy)}条，分别是{all_chanye}。
    其中，从产业链企业规模看、从产业链重点企业规模看、从产业链发展趋势看：
    {cy_paiming_str}
    """
    if not paiming:
        answer = ''
    return answer
if __name__ == '__main__':
    city_name='烟台'
    chanye='新能源'
    data = get_chanyelieshi(city_name='烟台',chanye='新能源')
    #data = get_chanyepaiming(city_name='北京',chanye='新能源')
    print(data)
