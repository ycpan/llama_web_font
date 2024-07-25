# -*- coding: utf-8 -*-

import threading
import requests
import json

access_token = "fk217408-4KdxNeEDSjmll43jQ0ItKVKmjhkvi7xH"

def get_hanyu_llm(prompt_text):
    #api_endpoint = "http://127.0.0.1:17860/api/completions"
    api_endpoint = "http://127.0.0.1:17870/api/completions"
    message = prompt_text
    if not isinstance(prompt_text,list):
        message = [{"role": "user", "content": prompt_text}]
    params = {
      #"model": "gpt-3.5-turbo",
      "model": "gpt-3.5-turbo-0613",
      #"model": "gpt4",
      #"messages": [{"role": "user", "content": "Hello!"}]
      #"messages": [{"role": "user", "content": prompt_text}],
      "messages": message,
      #"stream":True
      "stream":False
    }
    
    
    # Send the API request
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"}
    response = requests.post(api_endpoint, headers=headers, json=params)
    
    # Process the API response
    response_text = ''
    #print('QW:{}'.format(prompt_text))
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        #response_text = json.loads(response.text)["choices"][0]['message']['content']
        response_text = json.loads(response.text)
        #print(f"ChatGPT response: {response_text}")
        #import ipdb
        #ipdb.set_trace()
        return response_text['response']
    else:
        print(f"Error: {response.status_code} - {response.text}")
    return response_text
def get_hanyu_openai(prompt_text):
    api_endpoint = "http://127.0.0.1:8000/v1/chat/completions"
    message = prompt_text
    if not isinstance(prompt_text,list):
        message = [{"role": "user", "content": prompt_text}]
    params = {
      #"model": "gpt-3.5-turbo",
      "model": "gpt-3.5-turbo-0613",
      #"model": "gpt4",
      #"messages": [{"role": "user", "content": "Hello!"}]
      #"messages": [{"role": "user", "content": prompt_text}],
      "messages": message,
      #"stream":True
      "stream":False
    }
    
    
    # Send the API request
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"}
    response = requests.post(api_endpoint, headers=headers, json=params)
    
    # Process the API response
    response_text = ''
    #print('QW:{}'.format(prompt_text))
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        response_text = json.loads(response.text)["choices"][0]['message']['content']
        #response_text = json.loads(response.text)
        #print(f"ChatGPT response: {response_text}")
        #import ipdb
        #ipdb.set_trace()
        return response_text
    else:
        print(f"Error: {response.status_code} - {response.text}")
    return response_text
def send_post_request(url, payload):
    """
    向指定的URL发送POST请求。
    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    updated_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": payload["prompt"]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "string",
                    "description": "string",
                    "parameters": {}
                }
            }
        ],
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(updated_payload))
    try:
        response_json = response.json()
        print(response_json)
    except ValueError:
        print("Response could not be decoded as JSON:", response.text)


def threaded_requests(url, payload, num_threads, total_requests):
    """
    创建并启动多线程以达到指定的请求总量。
    """
    rounds = (total_requests + num_threads - 1) // num_threads  # 计算需要的轮数
    for _ in range(rounds):
        threads = []
        for _ in range(num_threads):
            if total_requests <= 0:
                break  # 如果已经达到请求总量，停止创建新线程
            #thread = threading.Thread(target=send_post_request, args=(url, payload))
            #insturction = "你的名字叫友谅科技智能助手，由北京友谅科技有限公司开发，请用专业的知识回答下面的问题。\n\n"
            payload = '杨金花与马龙与2013年9月份办理了结婚登记，婚后生育两子，长子马嘉鑫（出生于2008.1.13），长女马佳琪（出生2004.9.16）两人经常因家庭琐事发生矛盾纠纷已长达三年之久，先夫妻感情破裂，无法再继续生活。现申请新庄集乡人民调解委员调解处理。'
            #payload = insturction + payload
            #thread = threading.Thread(target=get_hanyu_openai, args=('我替她还16万元外债，离婚协议写的是10万元给子女的扶养费，可是说是还我的10元债务款我可以在起诉她吗？?'))
            thread = threading.Thread(target=get_hanyu_llm, args=(payload,))
            thread.start()
            threads.append(thread)
            total_requests -= 1

        for thread in threads:
            thread.join()


if __name__ == '__main__':
    api_url = 'http://127.0.0.1:8000/v1/chat/completions'
    payload = {
        "prompt": "解释一下量子计算"
    }
    num_threads = 50       # 线程数
    total_requests = 100   # 总请求数
    #num_threads = 1       # 线程数
    #total_requests = 1   # 总请求数

    threaded_requests(api_url, payload, num_threads, total_requests)
