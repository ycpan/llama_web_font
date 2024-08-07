import requests
import json
import pickle
import os 
import pandas as pd
#from .location_mapper import get_intergrity_from_str
#from location_mapper import get_intergrity_from_str

def get_intergrity_from_str(input_sentence):
    host = '10.0.0.16'
    port = '6666'
    post_data = {"text": input_sentence}
    docano_json = json.dumps(post_data,ensure_ascii=False)
    r = requests.post("http://"+host+":"+port+"/get_intergrity", json=docano_json)
    #print(r)
    result = r.text
    result = json.loads(result)
    province,city = result[0:2]
    return province,city
def get_chanye2code():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #index  = dir_path.find('/web')
    #web_path = dir_path[0:index + 5]
    #csv_path = os.path.join(web_path,'plugins_chanyeku/chanye2code.csv')
    csv_path = os.path.join(dir_path,'chanye2code.csv')
    data = pd.read_csv(csv_path)
    return data
#f = open('/devdata/home/user/panyongcan/Project/llama_web_font/web/plugins_chanyeku/industry2code.pkl','rb')
#industry2code = pickle.load(f)
industrycode = {}
data = get_chanye2code()
for _,da in data.iterrows():
    code = da['node']
    #if len(code) != 3:
    #    continue
    chanye = da['name']
    #if '新能源' in chanye:
    #    import ipdb
    #    ipdb.set_trace()
    industrycode[chanye.replace('产业','')]=code
    #industrycode[sub_key]=industry2code[key][sub_key]
    #industrycode[sub_key.replace('产业','')]=industry2code[key][sub_key]
#for key in industry2code:
#    for sub_key in industry2code[key]:
#        industrycode[sub_key]=industry2code[key][sub_key]
#        industrycode[sub_key.replace('产业','')]=industry2code[key][sub_key]
def get_chanyecode(chanye_name):
    if chanye_name in industrycode:
        return industrycode[chanye_name]
    #else:
    #    for key in industrycode:
    #        if chanye_name in key:
    #            return industrycode[key]
    return None
#url = "http://10.0.0.16:6092/enterprise_introduction"
url = "http://10.0.0.16:6093/enterprise_introduction"
def get_qiyeyinru(city_name='烟台',chanye=''):
    #import ipdb
    #ipdb.set_trace()
    #if chanye:
    #    chanye_code=industrycode[chanye]
    #else:
    #    chanye_code = ''
    chanye_code = get_chanyecode(chanye)
    if not chanye_code:
        return ''
    province,city = get_intergrity_from_str(city_name)
    if not city:
        return ''
    if province in '北京市天津市重庆市上海市':
        city = province
    payload = json.dumps({
      "industryCode": chanye_code,
      #"parentAreaName": "河南省",
      #"areaName": "郑州市",
      "province":province,
      "companyLevel":1,
      "city":city,
      "area":"",
      "parentAreaName": "sh",
      "resourceEndowment": [
        "年末实有城市道路面积 (万平方米)",
        "排水管道长度 (公里)",
        "境内公路总里程 (公里)",
        "高速公路里程 (公里)",
        "公路客运量 (万人)",
        "公路货运量 (万吨)",
        "供气总量(煤气、天然气) (万立方米)",
        "液化石油气供气总量 (吨)",
        "工业企业数",
        "港、澳、台商投资企业",
        "外商投资企业",
        "流动资产合计",
        "利润总额",
        "城镇非私营单位在岗职工平均工资",
        "普通高等学校数量",
        "中等职业教育学校数量",
        "普通高等学校专任教师数",
        "中等职业教育学校专任教师数",
        "普通本专科在校学生数",
        "中等职业教育学校在校学生数",
        "地区生产总值（当年价格）（亿元）",
        "人均地区生产总值 (元)",
        "地区生产总值增长率 (%) GRP",
        "地方一般公共预算收入",
        "地方一般公共预算支出",
        "科学技术支出",
        "年末金融机构人民币各项存款余额",
        "年末金融机构人民币各项贷款余额",
        "货物进口额",
        "货物出口额",
        "建成区面积 (平方公里)",
        "水资源总量（亿立方米）",
        "城市建设用地面积",
        "户籍人口",
        "医院数 (个)",
        "医院床位数 (张)",
        "养老机构数 (个)",
        "养老机构床位数 (张)",
        "邮政业务收入 (万元)",
        "电信业务收入 (万元)",
        "互联网宽带接入用户数 (万户)"
    ],
      "companyLevel": 1
    })
    payload1 = json.dumps({"resourceEndowment":
    ["工业企业数",
    "港、澳、台商投资企业",
    "外商投资企业",
    "流动资产合计",
    "利润总额",
    "城镇非私营单位在岗职工平均工资",
    "普通高等学校数量",
    "中等职业教育学校数量",
    "普通高等学校专任教师数",
    "普通本专科在校学生数",
    "中等职业教育学校在校学生数"],
    "province":"北京市",
    "companyLevel":1,
    "city":"北京市",
    "area":"西城区",
    "parentAreaName":"sh",
    "industryCode":"A01"})
    headers = {
      'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    data = ""
    try:
        data = json.loads(response.text)
    except:
        return ""
    company = [x['name'] for x in data['allCompany']['rows']]
    answer = f'{city_name}在{chanye}上应该引入的企业为:\n' + '\n'.join(company[0:10])
    if not company:
        answer = ''
    return answer
if __name__ == '__main__':
    #data = get_qiyeyinru('')
    #data = get_qiyeyinru('郑州市',chanye='半导体与集成电路产业')
    data = get_qiyeyinru('烟台',chanye='新能源')
    #data = get_qiyeyinru('河南',chanye='新能源')
    #data =get_qiyeyinru('岳阳县','补益药')
    print(data)
