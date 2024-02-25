import os
import openai
import requests
import json
import openai
import traceback
from retry import retry
from websocket import create_connection
from plugins.common import settings

#api_endpoint = "http://10.0.0.12:19327/v1/completions"
#api_endpoint = "http://10.0.0.12:8000/v1/completions"
access_token = "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"

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"

def generate_prompt(instruction, response="", with_system_prompt=True, system_prompt=None):
    if with_system_prompt is True:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': system_prompt})
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({'instruction': instruction})
    if len(response)>0:
        prompt += " " + response
    return prompt

def generate_completion_prompt(instruction: str):
    """Generate prompt for completion"""
    return generate_prompt(instruction, response="", with_system_prompt=True)


def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion"""

    system_msg = None
    for msg in messages:
        #if msg.role == 'system':
        if msg['role'] == 'system':
            #system_msg = msg.content
            system_msg = msg['content']
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg['role'] == 'system':
            continue
        if msg['role'] == 'user':
            if is_first_user_content is True:
                #prompt += generate_prompt(msg.content, with_system_prompt=True, system_prompt=system_msg)
                prompt += generate_prompt(msg['content'], with_system_prompt=True, system_prompt=system_msg)
                is_first_user_content = False
            else:
                #prompt += '<s>'+generate_prompt(msg.content, with_system_prompt=False)
                prompt += '<s>'+generate_prompt(msg['content'], with_system_prompt=False)
        if msg['role'] == 'assistant' or msg['role'] == 'ai':
                #prompt += f" {msg.content}"+"</s>"
                prompt += f" {msg['content']}"+"</s>"
    return prompt
def get_generate_prompt(question):
    if isinstance(question, str):
        prompt = generate_completion_prompt(question)
    else:
        prompt = generate_chat_prompt(question)
    return prompt
def get_dev_agent_output(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    api_endpoint = "http://10.0.0.20:19327/v1/completions"
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
def get_dev_llm_output(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    api_endpoint = "http://10.0.0.20:19328/v1/completions"
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
def get_dev_agent_output_fast(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    api_endpoint = "http://10.0.0.20:23336/v1/chat/completions"
    #api_endpoint = "http://10.0.0.12:23336/v1/chat/completions"
    input_messages = get_generate_prompt(input_str)
    #if isinstance(input_str,str):
    #    history_data = [ {"role": "system", "content": "You are a helpful assistant. 你是一个乐于助人的助手。\n"}]
    #    input_messages = [{"role": "user", "content": input_str}]
    #    history_data.extend(input_messages)
    #    input_messages = history_data

    params = {
      #"model": "gpt-3.5-turbo",
      #"model": "gpt-3.5-turbo-0613",
      #"model": "internlm2-chat-7b",
      "model": "llama2",
      "max_tokens":1512,
      #"temperature":0.2,
      "temperature":0.5,
      "repetition_penalty":1.1,
      "top_p":0.9,
      "top_k":40,
      #"model": "gpt-4-0613",
      #"messages": [{"role": "user", "content": "Hello!"}]
      #"messages": [{"role": "user", "content": input_str}],
      "messages": input_messages,
      #"stream":True
      "stream":False
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=params)
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        response_text = json.loads(response.text)["choices"][0]['message']['content']
    else:
        response_text = ''
    return response_text
def get_prod_agent_output_fast(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    #api_endpoint = "http://10.0.0.20:23336/v1/chat/completions"
    api_endpoint = "http://10.0.0.12:23336/v1/chat/completions"
    input_messages = get_generate_prompt(input_str)
    #if isinstance(input_str,str):
    #    history_data = [ {"role": "system", "content": "You are a helpful assistant. 你是一个乐于助人的助手。\n"}]
    #    input_messages = [{"role": "user", "content": input_str}]
    #    history_data.extend(input_messages)
    #    input_messages = history_data

    params = {
      #"model": "gpt-3.5-turbo",
      #"model": "gpt-3.5-turbo-0613",
      #"model": "internlm2-chat-7b",
      "model": "llama2",
      "max_tokens":1512,
      #"temperature":0.2,
      "temperature":0.5,
      "repetition_penalty":1.1,
      "top_p":0.9,
      "top_k":40,
      #"model": "gpt-4-0613",
      #"messages": [{"role": "user", "content": "Hello!"}]
      #"messages": [{"role": "user", "content": input_str}],
      "messages": input_messages,
      #"stream":True
      "stream":False
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=params)
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        response_text = json.loads(response.text)["choices"][0]['message']['content']
    else:
        response_text = ''
    return response_text
def get_zhishiku_output_fast(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    api_endpoint = "http://10.0.0.20:8000/v1/chat/completions"

    params = {
      #"model": "gpt-3.5-turbo",
      #"model": "gpt-3.5-turbo-0613",
      #"model": "internlm2-chat-7b",
      #"model": "llama2",
      "model": "/home/user/panyongcan/project/llm/origin/Orion-master/quantization/quantized_model",
      "max_tokens":2512,
      "temperature":0.2,
      "repetition_penalty":1.1,
      "top_p":0.9,
      "top_k":40,
      #"model": "gpt-4-0613",
      #"messages": [{"role": "user", "content": "Hello!"}]
      "messages": [{"role": "user", "content": input_str}],
      #"messages": input_messages,
      #"stream":True
      "stream":False
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=params)
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        response_text = json.loads(response.text)["choices"][0]['message']['content']
    else:
        response_text = ''
    return response_text
def get_dev_llm_output_fast(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    api_endpoint = "http://10.0.0.20:23333/v1/chat/completions"
    #api_endpoint = "http://10.0.0.20:8000/v1/chat/completions"
    #input_messages = { "prompt": input_str}
    input_messages = get_generate_prompt(input_str)
    #input_messages = [{"role": "user", "content": input_str}]
    params = {
      #"model": "gpt-3.5-turbo",
      #"model": "gpt-3.5-turbo-0613",
      "model": "internlm2-chat-7b",
      #"model": "llama2",
      #"model": "/home/user/panyongcan/project/llm/Chinese-LLaMA-Alpaca-2/scripts/Llama2-chat-13b",
      "max_tokens":1512,
      "temperature":0.2,
      "repetition_penalty":1.15,
      "top_p":0.9,
      "top_k":40,
      #"model": "gpt-4-0613",
      #"messages": [{"role": "user", "content": "Hello!"}]
      #"messages": [{"role": "user", "content": input_str}],
      "messages": input_messages,
      #"stream":True
      "stream":False
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=params)
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        response_text = json.loads(response.text)["choices"][0]['message']['content']
    else:
        response_text = ''
    return response_text
def get_prod_llm_output_fast(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    #api_endpoint = "http://10.0.0.12:23333/v1/completions"
    api_endpoint = "http://10.0.0.12:23333/v1/chat/completions"
    #input_messages = { "prompt": input_str}
    input_messages = get_generate_prompt(input_str)
    params = {
      #"model": "gpt-3.5-turbo",
      #"model": "gpt-3.5-turbo-0613",
      "model": "internlm2-chat-7b",
      #"messages": [{"role": "user", "content": input_str}],
      "messages": input_messages,
      #"stream":True
      "stream":False
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=params)
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        response_text = json.loads(response.text)["choices"][0]['message']['content']
    else:
        response_text = ''
    return response_text
def get_output_v1(input_sentence):
    """
    这个接口是测试小参数模型用的，不支持steam模式
    """
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

def get_prod_agent(data):
    """
    这个接口支持ws，但不支持stream模式
    """
    ws = create_connection("ws://127.0.0.1:"+str(17861)+"/ws")
    if isinstance(data,str):
        data = {'prompt':data,'history':[]}
    if isinstance(data,list):
        data = {'prompt':data,'history':[]}
    ws.send(json.dumps(data))
    #response.content_type = "application/json"
    temp_result = ''
    try:
        while True:
            result = ws.recv()
            if len(result) > 0:
                temp_result = result
    except Exception as e:
        #print(e)
        pass
    ws.close()
    #data = json.dumps({"response": temp_result},ensure_ascii=False)
    return temp_result
def get_prod_llm(data):
    """
    这个接口支持ws，但不支持stream模式
    """
    ws = create_connection("ws://127.0.0.1:"+str(17862)+"/ws")
    if isinstance(data,str):
        data = {'prompt':data,'history':[]}
    if isinstance(data,list):
        data = {'prompt':data,'history':[]}
    ws.send(json.dumps(data))
    #response.content_type = "application/json"
    temp_result = ''
    try:
        while True:
            result = ws.recv()
            if len(result) > 0:
                temp_result = result
    except Exception as e:
        #print(e)
        pass
    ws.close()
    #data = json.dumps({"response": temp_result},ensure_ascii=False)
    return temp_result

def get_prod_stream_agent(data):
    """
    这个接口支持ws，但是能支持stream模式
    """
    ws = create_connection("ws://127.0.0.1:"+str(17862)+"/ws")
    if isinstance(data,str):
        data = {'prompt':data,'history':[]}
    if isinstance(data,list):
        data = {'prompt':data,'history':[]}
    ws.send(json.dumps(data))
    #import ipdb
    #ipdb.set_trace()
    try:
        is_generate_normal = True
        while True:
            result = ws.recv()
            if len(result) > 0:
                #yield "data: %s\n\n" % json.dumps({"response": result})
                if '</' in result or '[INST]' in result or '<s>' in result:
                    #break
                    is_generate_normal = False
                if is_generate_normal:
                    yield "%s\n\n" % json.dumps({"response": result},ensure_ascii=False)
    except Exception as e:
        print(e)
        #import ipdb
        #ipdb.set_trace()
        pass
    ws.close()
    yield "data: %s\n\n" % "[DONE]"
def get_prod_stream_llm(data):
    """
    这个接口支持ws，但是能支持stream模式
    """
    ws = create_connection("ws://127.0.0.1:"+str(17862)+"/ws")
    if isinstance(data,str):
        data = {'prompt':data,'history':[]}
    if isinstance(data,list):
        data = {'prompt':data,'history':[]}
    ws.send(json.dumps(data))
    #import ipdb
    #ipdb.set_trace()
    try:
        is_generate_normal = True
        while True:
            result = ws.recv()
            if len(result) > 0:
                #yield "data: %s\n\n" % json.dumps({"response": result})
                if '</' in result or '[INST]' in result or '<s>' in result:
                    #break
                    is_generate_normal = False
                if is_generate_normal:
                    yield "%s\n\n" % json.dumps({"response": result},ensure_ascii=False)
    except Exception as e:
        print(e)
        #import ipdb
        #ipdb.set_trace()
        pass
    ws.close()
    yield "data: %s\n\n" % "[DONE]"
# delay 表示延迟1s再试；backoff表示每延迟一次，增加2s，max_delay表示增加到120s就不再增加； tries=3表示最多试3次
@retry(delay=8, backoff=4, max_delay=22,tries=2)
def completion_with_backoff(**kwargs):
    try:
        return  openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        #import ipdb
        #ipdb.set_trace()
        print(e)

        #import ipdb
        #ipdb.set_trace()
        dm = DelataMessage()
        if "maximum context length is 8192 tokens" in str(e):
            print('maximum exceed,deal ok')
            content= '历史记录过多，超过规定长度，请清空历史记录'
            setattr(dm,'content',content)
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        if "Name or service not known" in str(e):
            print('域名设置有问题，请排查服务器域名')
            content = '域名设置有问题，请排查服务器域名'
            setattr(dm,'content',content)
            #chunk=[{'choices':[{'finish_reason':'continue','delta':{'content':dm}}]}]
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        raise e  
def get_output_with_openai(history_data):
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    openai.api_base = settings.llm.api_host
    response = completion_with_backoff(model="internlm2-chat-7b", messages=history_data, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    resTemp = ''
    try:
        for chunk in response:
            #print(chunk)
            if chunk['choices'][0]["finish_reason"]!="stop":
                if hasattr(chunk['choices'][0]['delta'], 'content'):
                    resTemp+=chunk['choices'][0]['delta']['content']
                    #yield resTemp
                    yield "%s\n\n" % json.dumps({"response": resTemp},ensure_ascii=False)
        #return resTemp
    except:
        traceback.print_exc()
        pass
    yield "data: %s\n\n" % "[DONE]"


class DelataMessage:
    def __init__(self):
        self.content=''
    def __getitem__(self, item):
        return getattr(self, item)
# delay 表示延迟1s再试；backoff表示每延迟一次，增加2s，max_delay表示增加到120s就不再增加； tries=3表示最多试3次
@retry(delay=8, backoff=4, max_delay=22,tries=2)
def completion_with_backoff(**kwargs):
    try:
        return  openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        #import ipdb
        #ipdb.set_trace()
        print(e)

        #import ipdb
        #ipdb.set_trace()
        dm = DelataMessage()
        if "maximum context length is 8192 tokens" in str(e):
            print('maximum exceed,deal ok')
            content= '历史记录过多，超过规定长度，请清空历史记录'
            setattr(dm,'content',content)
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        if "Name or service not known" in str(e):
            print('域名设置有问题，请排查服务器域名')
            content = '域名设置有问题，请排查服务器域名'
            setattr(dm,'content',content)
            #chunk=[{'choices':[{'finish_reason':'continue','delta':{'content':dm}}]}]
            chunk=[{'choices':[{'finish_reason':'continue','delta':dm,'content':content}]}]
            return chunk
        raise e  
def get_dev_stream_with_openapi(data):
    #import ipdb
    #ipdb.set_trace()
    #history_data = [ {"role": "system", "content": "You are a helpful assistant."}]
    #import ipdb
    #ipdb.set_trace()
    #input_messages = [{"role": "user", "content": data}]
    #if isinstance(data,list):
    #    input_messages = data
    #import ipdb
    #ipdb.set_trace()
    input_messages = get_generate_prompt(data)
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    api_endpoint = "http://10.0.0.20:23333/v1"
    #api_endpoint = "http://10.0.0.20:8000/v1"
    openai.api_base = api_endpoint
    #openai.api_base = settings.llm.api_host
    #response = completion_with_backoff(model="internlm2-chat-7b", temperature=0.6,repetition_penalty=1.08,top_p=0.7,top_k=20,messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    response = completion_with_backoff(model="internlm2-chat-7b", temperature=0.2,repetition_penalty=1.08,top_p=0.6,top_k=40,messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    #response = completion_with_backoff(model="internlm2-chat-7b", temperature=0.6,repetition_penalty=1.02,top_p=0.6,top_k=40,messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    #response = completion_with_backoff(model="internlm2-chat-7b", temperature=0.6,repetition_penalty=1.1,top_p=0.6,top_k=40,messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    #response = completion_with_backoff(model="/home/user/panyongcan/project/llm/Chinese-LLaMA-Alpaca-2/scripts/Llama2-chat-13b", temperature=0.6,repetition_penalty=1.35,top_p=0.6,top_k=20,messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    resTemp=""
    try:
        for chunk in response:
            #print(chunk)
            if chunk['choices'][0]["finish_reason"]!="stop":
                if hasattr(chunk['choices'][0]['delta'], 'content'):
                    resTemp+=chunk['choices'][0]['delta']['content']
                    #yield resTemp
                    yield "%s\n\n" % json.dumps({"response": resTemp},ensure_ascii=False)
        #return resTemp
    except:
        traceback.print_exc()
        pass
    yield "data: %s\n\n" % "[DONE]"
def get_prod_stream_with_openapi(data):
    #import ipdb
    #ipdb.set_trace()
    #history_data = [ {"role": "system", "content": "You are a helpful assistant."}]
    input_messages = get_generate_prompt(data)
    #input_messages = [{"role": "user", "content": data}]
    #if isinstance(data,list):
    #    input_messages = data
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    api_endpoint = "http://10.0.0.12:23333/v1"
    openai.api_base = api_endpoint
    #openai.api_base = settings.llm.api_host
    #response = completion_with_backoff(model="internlm2-chat-7b", messages=history_data, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    #response = completion_with_backoff(model="internlm2-chat-7b", messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    response = completion_with_backoff(model="internlm2-chat-7b", temperature=0.6,repetition_penalty=1.1,top_p=0.6,top_k=40,messages=input_messages, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    resTemp=""
    try:
        for chunk in response:
            #print(chunk)
            if chunk['choices'][0]["finish_reason"]!="stop":
                if hasattr(chunk['choices'][0]['delta'], 'content'):
                    resTemp+=chunk['choices'][0]['delta']['content']
                    #yield resTemp
                    yield "%s\n\n" % json.dumps({"response": resTemp},ensure_ascii=False)
        #return resTemp
    except:
        traceback.print_exc()
        pass
    yield "data: %s\n\n" % "[DONE]"
def get_zhishiku_stream_with_openapi(data):
    if isinstance(data,str):
        data = [{"role": "user", "content": data}]
    #history_data = [ {"role": "system", "content": "You are a helpful assistant."}]
    #input_messages = get_generate_prompt(data)
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    api_endpoint = "http://10.0.0.20:8000/v1"
    openai.api_base = api_endpoint
    model = "/home/user/panyongcan/project/llm/origin/Orion-master/quantization/quantized_model"
    response = completion_with_backoff(model=model, temperature=0.2,repetition_penalty=1.1,top_p=0.9,top_k=40,max_tokens=2048,messages=data,  stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
    resTemp=""
    try:
        for chunk in response:
            #print(chunk)
            if chunk['choices'][0]["finish_reason"]!="stop":
                if hasattr(chunk['choices'][0]['delta'], 'content'):
                    resTemp+=chunk['choices'][0]['delta']['content']
                    #yield resTemp
                    yield "%s\n\n" % json.dumps({"response": resTemp},ensure_ascii=False)
        #return resTemp
    except:
        traceback.print_exc()
        pass
    yield "data: %s\n\n" % "[DONE]"
