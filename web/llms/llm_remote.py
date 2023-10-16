import requests
import json

#api_endpoint = "http://10.0.0.20:19327/v1/completions"
api_endpoint = "http://10.0.0.12:19327/v1/completions"
#api_endpoint = "http://10.0.0.12:8000/v1/completions"
access_token = "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"

def get_output(input_str):
    input_messages = { "prompt": input_str}
    headers = {"Content-Type": "application/json",
               #"Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=input_messages)
    if response.status_code == 200:
        response_text = json.loads(response.text)["choices"][0]["text"]
    else:
        response_text = ''
    return response_text
def get_output_v1(input_sentence):
    #input_sentence = '南京有什么好玩的？'
    #input_sentence = '北京市专精特新企业列表'
   
    host = '10.0.0.20'
    port = '6666'
    post_data = {"text": input_sentence}
    docano_json = json.dumps(post_data,ensure_ascii=False)
    r = requests.post("http://"+host+":"+port+"/run", json=docano_json)
    #print(r)
    result = r.text
    return result
