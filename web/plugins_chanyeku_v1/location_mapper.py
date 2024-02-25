import requests
import json

host = '10.0.0.16'
#port = '10001'
port = '6666'
#for i in range(1,5):
#    input_sentence = [
#            '铅山地处江西东北部，南越武夷山与福建毗邻，华东第一峰——黄岗山穿界而过，北临信江，东近浙江。黄岗山玉绿地域保护范围包括武夷山镇、英将乡、鹅湖镇鹅湖村、太源乡、篁碧乡、天柱山乡等6个乡（镇）。保护范围位于东经117°44′33〞-117°47′31〞，北纬27°48′19〞-28°17′16〞之间，保护面积1333公顷，年产量300吨。',
#            ]
def get_code_from_str(input_sentence):
    host = '10.0.0.16'
    port = '6666'
    post_data = {"text": input_sentence}
    docano_json = json.dumps(post_data,ensure_ascii=False)
    r = requests.post("http://"+host+":"+port+"/get_code", json=docano_json)
    #print(r)
    result = r.text
    #print(result)
    return result
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
