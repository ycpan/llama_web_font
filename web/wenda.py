from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import time
from starlette.requests import Request
from starlette.responses import FileResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import functools
import bottle
from bottle import route, response, request, static_file, hook
import datetime
import json
import os
import threading
import torch
from plugins.common import error_helper, error_print, success_print
from plugins.common import allowCROS
from plugins.common import settings
from plugins.common import app
import logging
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import aiomysql
import re
logging.captureWarnings(True)
logger = None
try:
    from loguru import logger
except:
    pass


def load_LLM():
    try:
        from importlib import import_module
        LLM = import_module('llms.llm_'+settings.llm_type)
        return LLM
    except Exception as e:
        logger and logger.exception(e)
        print("LLM模型加载失败，请阅读说明：https://github.com/l15y/wenda", e)


LLM = load_LLM()

logging = settings.logging
if logging:
    from plugins.defineSQL import session_maker, 记录


model = None
tokenizer = None


def load_model():
    LLM.load_model()
    torch.cuda.empty_cache()
    success_print("模型加载完成")


if __name__ == '__main__':
    thread_load_model = threading.Thread(target=load_model)
    thread_load_model.start()
zhishiku = None


def load_zsk():
    try:
        global zhishiku
        import plugins.zhishiku as zsk
        zhishiku = zsk
        success_print("知识库加载完成")
    except Exception as e:
        logger and logger.exception(e)
        error_helper(
            "知识库加载失败，请阅读说明", r"https://github.com/l15y/wenda#%E7%9F%A5%E8%AF%86%E5%BA%93")
        raise e


if __name__ == '__main__':
    thread_load_zsk = threading.Thread(target=load_zsk)
    thread_load_zsk.start()

chanyeku = None


def load_zsk():
    try:
        global chanyeku
        import plugins_chanyeku.chanyeku as cyk
        chanyeku = cyk
        success_print("产业库加载完成")
    except Exception as e:
        error_helper(
            "知识库加载失败，请阅读说明", r"https://www.incoshare.com/#%E7%9F%A5%E8%AF%86%E5%BA%93")
        raise e


if __name__ == '__main__':
    thread_load_zsk = threading.Thread(target=load_zsk)
    thread_load_zsk.start()

@route('/llm')
def llm_js():
    noCache()
    return static_file('llm_'+settings.llm_type+".js", root="llms")


@route('/plugins')
def read_auto_plugins():
    noCache()
    plugins = []
    for root, dirs, files in os.walk("autos"):
        for file in files:
            if(file.endswith(".js")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    plugins.append({"name": file, "content": f.read()})
    return json.dumps(plugins)
# @route('/writexml', method=("POST","OPTIONS"))
# def writexml():
    # data = request.json
    # s=json2xml(data).decode("utf-8")
    # with open(os.environ['wenda_'+'Config']+"_",'w',encoding = "utf-8") as f:
    #     f.write(s)
    #     # print(j)
    #     return s


def noCache():
    response.set_header("Pragma", "no-cache")
    response.add_header("Cache-Control", "must-revalidate")
    response.add_header("Cache-Control", "no-cache")
    response.add_header("Cache-Control", "no-store")


def pathinfo_adjust_wrapper(func):
    # A wrapper for _handle() method
    @functools.wraps(func)
    def _(s, environ):
        environ["PATH_INFO"] = environ["PATH_INFO"].encode(
            "utf8").decode("latin1")
        return func(s, environ)
    return _


bottle.Bottle._handle = pathinfo_adjust_wrapper(
    bottle.Bottle._handle)  # 修复bottle在处理utf8 url时的bug


@hook('before_request')
def validate():
    REQUEST_METHOD = request.environ.get('REQUEST_METHOD')
    HTTP_ACCESS_CONTROL_REQUEST_METHOD = request.environ.get(
        'HTTP_ACCESS_CONTROL_REQUEST_METHOD')
    if REQUEST_METHOD == 'OPTIONS' and HTTP_ACCESS_CONTROL_REQUEST_METHOD:
        request.environ['REQUEST_METHOD'] = HTTP_ACCESS_CONTROL_REQUEST_METHOD


waiting_threads = 0


@route('/chat_now', method=('GET', "OPTIONS"))
def api_chat_now():
    allowCROS()
    noCache()
    return {'queue_length': waiting_threads}


@route('/find', method=("POST", "OPTIONS"))
def api_find():
    allowCROS()
    data = request.json
    if not data:
        return '0'
    prompt = data.get('prompt')
    step = data.get('step')
    if step is None:
        step = int(settings.library.step)
    return json.dumps(zhishiku.find(prompt, int(step)))


@route('/completions', method=("POST", "OPTIONS"))
def api_chat_box():
    response.content_type = "text/event-stream"
    response.add_header("Connection", "keep-alive")
    response.add_header("Cache-Control", "no-cache")
    response.add_header("X-Accel-Buffering", "no")
    data = request.json
    #import ipdb
    #ipdb.set_trace()
    messages = data.get('messages')
    stream = data.get('stream')
    prompt = messages[-1]['content']
    data['prompt'] = prompt
    history = []
    for i, old_chat in enumerate(messages[0:len(messages)-1]):
        if old_chat['role'] == "user":
            history.append(old_chat)
        elif old_chat['role'] == "assistant":
            old_chat['role'] = "AI"
            history.append(old_chat)
        else:
            continue
    data['history'] = history
    data['level'] = 0
    from websocket import create_connection
    ws = create_connection("ws://127.0.0.1:"+str(settings.port)+"/ws")
    ws.send(json.dumps(data))
    if not stream:
        response.content_type = "application/json"
        temp_result = ''
        try:
            while True:
                result = ws.recv()
                if len(result) > 0:
                    temp_result = result
        except:
            pass
        yield json.dumps({"response": temp_result})

    else:
        try:
            while True:
                result = ws.recv()
                if len(result) > 0:
                    yield "data: %s\n\n" % json.dumps({"response": result})
        except:
            pass
        yield "data: %s\n\n" % "[DONE]"

    ws.close()


@route('/chat_stream', method=("POST", "OPTIONS"))
def api_chat_stream():
    allowCROS()
    response.add_header("Connection", "keep-alive")
    response.add_header("Cache-Control", "no-cache")
    response.add_header("X-Accel-Buffering", "no")
    data = request.json
    data = json.dumps(data)
    from websocket import create_connection
    ws = create_connection("ws://127.0.0.1:"+str(settings.port)+"/ws")
    ws.send(data)
    try:
        while True:
            result = ws.recv()
            if len(result) > 0:
                yield result
    except:
        pass
    ws.close()


@route('/chat', method=("POST", "OPTIONS"))
def api_chat():
    allowCROS()
    data = request.json
    data = json.dumps(data)
    from websocket import create_connection
    ws = create_connection("ws://127.0.0.1:"+str(settings.port)+"/ws")
    ws.send(data)
    try:
        while True:
            new_result = ws.recv()
            if len(new_result) > 0:
                result = new_result
    except:
        pass
    ws.close()
    print([result])
    return result


bottle.debug(True)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):

    start_time = time.time()
    response = await call_next(request)
    path=request.scope['path']
    if path.startswith('/static/') and not path.endswith(".html"):
        return response
        
    process_time = time.time() - start_time
    response.headers["X-Process-Times"] = str(process_time)
    response.headers["Pragma"] = "no-cache"
    response.headers["Cache-Control"] = "no-cache,no-store,must-revalidate"

    return response
users_count = [0]*4


def get_user_count_before(level):
    count = 0
    for i in range(level):
        count += users_count[i]
    return count


class AsyncContextManager:
    def __init__(self, level):
        self.level = level

    async def __aenter__(self):
        users_count[self.level] += 1

    async def __aexit__(self, exc_type, exc, tb):
        users_count[self.level] -= 1


Lock = AsyncContextManager
def build_wechat_history(history_formatted):
    history_formatted = history_formatted[::-1]
    history_data = []
    if history_formatted is not None:
        for i, old_chat in enumerate(history_formatted):
            if 'question' in old_chat:
                #if old_chat['role'] == "user":
                history_data.append(
                    {"role": "user", "content": old_chat['question']})
            if 'answer' in old_chat:
                #if old_chat['role'] == "user":
                history_data.append(
                    #{"role": "user", "content": old_chat['question']})
                    #{"role": "assistant", "content": old_chat['content']},)
                    {"role": "assistant", "content": old_chat['answer']},)
                    #{"role": "AI", "content": old_chat['answer']},)
            #    elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
            #        if i > len(history_formatted) - 4:
            #            history_data.append(
            #                {"role": "assistant", "content": old_chat['content']},)
            #else:
            #    history_data.append({"role":"user","content":old_chat["question"]})
            #    history_data.append({"role":"assistant","content":old_chat["answer"]})
    return history_data
def build_web_history(history_formatted):
    history_data = []
    if history_formatted is not None:
        for i, old_chat in enumerate(history_formatted):
            #if 'question' in old_chat:
            if 'role' in old_chat:
                if old_chat['role'] == "user":
                    history_data.append(
                        {"role": "user", "content": old_chat['content']})
            #if 'answer' in old_chat:
            #    #if old_chat['role'] == "user":
            #    history_data.append(
            #        #{"role": "user", "content": old_chat['question']})
            #        #{"role": "assistant", "content": old_chat['content']},)
            #        {"role": "assistant", "content": old_chat['answer']},)
            #        #{"role": "AI", "content": old_chat['answer']},)
                elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                    #if i > len(history_formatted) - 4:
                    history_data.append(
                        {"role": "assistant", "content": old_chat['content']},)
            #else:
            #    history_data.append({"role":"user","content":old_chat["question"]})
            #    history_data.append({"role":"assistant","content":old_chat["answer"]})
    return history_data
def build_gov_history(history_formatted):
    history_data = []
    if history_formatted is not None:
        for i, old_chat in enumerate(history_formatted):
            #if 'question' in old_chat:
            if 'role' in old_chat:
                if old_chat['role'] == "user":
                    history_data.append(
                        {"role": "user", "content": old_chat['content']})
            #if 'answer' in old_chat:
            #    #if old_chat['role'] == "user":
            #    history_data.append(
            #        #{"role": "user", "content": old_chat['question']})
            #        #{"role": "assistant", "content": old_chat['content']},)
            #        {"role": "assistant", "content": old_chat['answer']},)
            #        #{"role": "AI", "content": old_chat['answer']},)
                elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                    #if i > len(history_formatted) - 4:
                    history_data.append(
                        {"role": "assistant", "content": old_chat['content']},)
            #else:
            #    history_data.append({"role":"user","content":old_chat["question"]})
            #    history_data.append({"role":"assistant","content":old_chat["answer"]})
    history_data = history_data[::-1]
    return history_data

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    global waiting_threads
    await websocket.accept()
    waiting_threads += 1
    # await asyncio.sleep(5)
    import base64
    try:
        data = await websocket.receive_json()
        #if 'task_type' in data and data['task_type'] == "transfile_file":
        #    index=data['content'].find(',')
        #    encoded_data = data['content'][index+1:]
        #    decoded_data = base64.b64decode(encoded_data)
        #    f = open('test.doc','wb')
        #    f.write(decoded_data)
        #    f.close()
        #    await websocket.send_text("学习已经完成")
        #    await websocket.close()
        #    return ""
        if 'task_type' in data and data['task_type'] == "transfile_file":
            index=data['content'].find(',')
            encoded_data = data['content'][index+1:]
            decoded_data = base64.b64decode(encoded_data)
            #import ipdb
            #ipdb.set_trace()
            f = open(f'./txt/test.{data["file_type"]}','wb')
            f.write(decoded_data)
            f.close()
            await websocket.send_text("学习已经完成")
            await websocket.close()
            return ""
        if 'task_type' in data:
            """
            gov 端
            """
            #import ipdb
            #ipdb.set_trace()
            data['history'] = build_gov_history(data['history'])
        # {'file_path': '', 'file_path_time': '', 'question': '北京市今年发展情况怎么样'}
        #{'file_path': '', 'file_path_time': '', 'question': '北京市今年发展情况详解', 'history': [{'question': '北京市今年发展情况详解', 'answer': "错误'role'"}, {'question': '北京市今年发展情况怎么样', 'answer': '错误'}]}
        elif 'question' in data:
            """
            小程序端
            """
            #import ipdb
            #ipdb.set_trace()
            if 'prompt' not in data:
                data['prompt'] = data['question']
            #if 'history' in data:
            #    #import ipdb
            #    #ipdb.set_trace()
            #    print(1)
            #    #data['history'] = []
            if 'history' in data:
                data['history'] = build_wechat_history(data['history'])
            else:
                data['history'] = []
        else:
            """
            web 端
            """
            #import ipdb
            #ipdb.set_trace()
            #data['history'] = build_web_history(data['history'])
            pass
            

        if 'history' not in data:
            data['history'] = []
        prompt = data.get('prompt')
        max_length = data.get('max_length')
        if max_length is None:
            max_length = 2048
        top_p = data.get('top_p')
        if top_p is None:
            top_p = 0.7
        temperature = data.get('temperature')
        if temperature is None:
            temperature = 0.9
        keyword = data.get('keyword')
        if keyword is None:
            keyword = prompt
        level = data.get('level')
        if level is None:
            level = 3
        history = data.get('history')
        history_formatted = LLM.chat_init(history)
        response = ''
        IP = websocket.client.host
        count_before = get_user_count_before(4)

        if count_before >= 4-level:
            time2sleep = (count_before+1)*level
            while time2sleep > 0:
                await websocket.send_text('正在排队，当前计算中用户数：'+str(count_before)+'\n剩余时间：'+str(time2sleep)+"秒")
                await asyncio.sleep(1)
                count_before = get_user_count_before(4)
                if count_before < 4-level:
                    break
                time2sleep -= 1
        lock = Lock(level)
        async with lock:
            if isinstance(prompt,str):
                print("\033[1;32m"+IP+":\033[1;31m"+prompt+"\033[1;37m")
            try:
                #for response in LLM.chat_one(prompt, history_formatted, max_length, top_p, temperature, data):
                text = ''
                for response in LLM.chat_one(prompt, history_formatted, max_length, top_p, temperature, data, zhishiku,chanyeku=chanyeku):
                    if (response):
                        # start = time.time()
                        text = response
                        await websocket.send_text(response)
                        await asyncio.sleep(0)
                        # end = time.time()
                        # cost+=end-start

                #await websocket.send_text(text + '\n<br /><br />\n<p style="color: red;">AI生成内容仅供参考</p>')
                await asyncio.sleep(0)
            except Exception as e:
                error = str(e)
                await websocket.send_text("错误"+ error)
                await websocket.close()
                raise e
            torch.cuda.empty_cache()
        if logging:
            with session_maker() as session:
                jl = 记录(时间=datetime.datetime.now(),
                        IP=IP, 问=prompt, 答=response)
                session.add(jl)
                session.commit()
        print(response)
        #websocket.send_text("[DONE]"+ error)
        await websocket.close()
    except WebSocketDisconnect:
        pass
    waiting_threads -= 1
@app.get("/download", summary="下载文件")
async def download_file(file_name):
    #data = request.json
    #file_name = '下面是烟台分布的图表.pdf'
    #file_path = os.path.join('./report',file_name)
    file_path = file_name
    if os.path.isfile(file_path):
        return FileResponse(file_path, filename=file_name)
    else:
        return f"请求的文件《{file_name}》不存在"

@app.get("/")
async def index(request: Request):
    return RedirectResponse(url="/index.html")
# MySQL数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'mysql@033471',
    'db': 'Algrithm',
    #'minsize': 1,
    #'maxsize': 10
}
# 初始化Jinja2模板引擎
templates = Jinja2Templates(directory="templates")
@app.get('/fatiao',response_class=HTMLResponse)
async def get_fatiao(request: Request, keyword: str = ""):
    # http://114.242.71.36:17870/fatiao?keyword=%E5%A9%9A
    #noCache()
    #keyword = request.get('keyword', '')
    results = ['nihao ya','这个是进行测试用的']
    results = []
    if keyword:
        conn = await aiomysql.connect(**DB_CONFIG)
        cur = await conn.cursor(aiomysql.DictCursor)
        await cur.execute("SELECT `法规名称`,`法条名称`,`法条内容` FROM Algrithm.4_法条推送_c类问答场景需要 WHERE `纠纷类型` LIKE %s", ('%' + keyword + '%',))
        results = await cur.fetchall()
        await cur.close()
        conn.close()
    #import ipdb
    #ipdb.set_trace()
    #return render_template('fatiao.html', keyword=keyword, results=results)
    return templates.TemplateResponse("fatiao.html", {"request": request, "keyword": keyword, "results": results})
    #return static_file('llm_'+settings.llm_type+".js", root="llms")
@app.get('/anli',response_class=HTMLResponse)
async def get_fatiao(request: Request, keyword: str = ""):
    # http://114.242.71.36:17870/fatiao?keyword=%E5%A9%9A
    #noCache()
    results = []
    if keyword:
        conn = await aiomysql.connect(**DB_CONFIG)
        cur = await conn.cursor(aiomysql.DictCursor)
        await cur.execute("SELECT `案例全文` FROM Algrithm.5_案例推送_c类问答场景需要 WHERE `纠纷类型` LIKE %s", ('%' + keyword + '%',))
        results = await cur.fetchall()
        await cur.close()
        conn.close()
    #import ipdb
    #ipdb.set_trace()
    return templates.TemplateResponse("anli.html", {"request": request, "keyword": keyword, "results": results})
def recormend_simility_question(prompt):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key="56e5dd2c23e1549751789deca7905ee5.TXYFSFrGpk9hG7vv") # 请填写您自己的APIKey
    #system = '你的名字叫友谅科技智能助手，由北京友谅科技有限公司开发，请用专业的知识回答下面的问题。'
    #new_history_data = [{'role':'system','content':system}]
    history_data = []
    new_prompt = '请根据下面的问题引申出可能要追问的的问题,并返回python list格式，list元素不超过10个:\n\n' + prompt
    history_data.append({"role": "user", "content": new_prompt})
    #new_history_data.extend(history_data)
    response = client.chat.completions.create(
    #response = client.chat.asyncCompletions.create(
    model="GLM-4-Flash",  # 填写需要调用的模型名称
    #messages=new_history_data,
    messages=history_data,
    stream=False,
    )
    
    #import ipdb
    #ipdb.set_trace()
    resp = response.choices[0].message.content
    #resp = response
    return resp
#def chat_one(prompt, history_formatted, max_length, top_p, temperature, data):
@app.get('/api/similar-questions')
async def get_fatiao(request: Request, question: str = ""):
    # http://114.242.71.36:17870/fatiao?question=%E5%A9%9A
    #noCache()
    results = []
    if question:
        #conn = await aiomysql.connect(**DB_CONFIG)
        #cur = await conn.cursor(aiomysql.DictCursor)
        #await cur.execute("SELECT `案例全文` FROM Algrithm.5_案例推送_c类问答场景需要 WHERE `纠纷类型` LIKE %s", ('%' + question + '%',))
        #results = await cur.fetchall()
        #await cur.close()
        #conn.close()
        results = recormend_simility_question(question)
        #import ipdb
        #ipdb.set_trace()
        try:
            clean_results = results
            if '```' in clean_results:
                clean_results = clean_results.replace('```','').strip('python\n')
                clean_results = re.sub('\d*\. ','',clean_results)
                results = eval(clean_results)
            else:
                clean_results = re.split('\d*\. ',clean_results)
                results = [x.strip() for x in clean_results if x.strip()]
            results = results[0:10]
        except:
            results =[]
    #import ipdb
    #ipdb.set_trace()
    #return templates.TemplateResponse("anli.html", {"request": request, "question": question, "results": results})
    return results
app.mount(path="/chat/", app=WSGIMiddleware(bottle.app[0]))
app.mount(path="/api/", app=WSGIMiddleware(bottle.app[0]))
app.mount("/txt/", StaticFiles(directory="txt"), name="txt")
app.mount("/", StaticFiles(directory="views"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port,
                log_level='error', loop="asyncio")
