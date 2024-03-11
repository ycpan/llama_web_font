import os
import time
import json
import copy
import numpy as np
import openai
import pandas as pd
from retry import retry
#from .llm_remote import get_dev_agent_output as get_agent
#from .llm_remote import get_prod_agent as get_agent
#from .llm_remote import get_dev_agent_output_fast as get_agent
from .llm_remote import get_prod_agent_output_fast as get_agent
#from .llm_remote import get_prod_stream_llm as get_stream_llm
#from .llm_remote import get_prod_llm as get_llm
#from .llm_remote import get_dev_llm_output as get_stream_llm
from .llm_remote import get_prod_stream_with_openapi as get_stream_llm
#from .llm_remote import get_dev_stream_with_openapi as get_stream_llm
#from .llm_remote import get_zhishiku_stream_with_openapi as get_zhishiku_stream_llm
#from .llm_remote import get_zhishiku_stream_with_openapi as get_stream_llm
#from .llm_remote import get_prod_stream_llm as get_stream_llm
#from .llm_remote import get_dev_llm_output as get_llm
from .llm_remote import get_prod_llm_output_fast as get_llm
from .llm_remote import get_prod_llm_output_fast as get_zhishiku_llm
##from .llm_remote import get_dev_llm_output_fast as get_llm
##from .llm_remote import get_dev_llm_output_fast as get_zhishiku_llm
#from .llm_remote import get_zhishiku_output_fast as get_zhishiku_llm
#from .llm_remote import get_prod_llm as get_llm
from plugins.common import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chat_init(history):
    return history

class DelataMessage:
    def __init__(self):
        self.content=''
    def __getitem__(self, item):
        return getattr(self, item)
def get_web_data(solution_prompt,zhishiku):
    #import ipdb
    #ipdb.set_trace()
    solution_data = zhishiku.zsk[1]['zsk'].find(solution_prompt)
    if not solution_data:
        solution_data = zhishiku.zsk[0]['zsk'].find(solution_prompt)
        if len(solution_data) > 0:
            zhishiku.zsk[1]['zsk'].save(solution_prompt,solution_prompt,solution_data,'','')
            print('save {} mysql successfully'.format(solution_prompt))
    if solution_data:
        return solution_data
    return ''
def exec_step(current_plan,zhishiku,chanyeku,current_bak_data=''):
    try:
        current_li = current_plan.split(':')
        solution_type= current_li[0]
        solution_exec = ','.join(current_li[1:])
        solution_exec = solution_exec.strip()
        #import ipdb
        #ipdb.set_trace()
        solution_bak_data = None
        if 'llm' in solution_type:
            new_prompt = str(current_bak_data) + ' ' + solution_exec
            #answer = get_llm(solution_exec)
            #answer = get_llm(new_prompt)
            answer = get_zhishiku_llm(new_prompt)
            solution_data = answer
            #solution_data_res.append({solution_exec:solution_data})
            if solution_data:
                solution_bak_data = str({solution_exec:solution_data})
                #solution_bak_data = solution_data
            #break
        if 'eval' in solution_type:
            #new_prompt = str(current_bak_data) + ' ' + solution_exec + '请回答是或否'
            ##answer = get_llm(solution_exec)
            #answer = get_llm(new_prompt)
            #solution_data = answer
            ##solution_data_res.append({solution_exec:solution_data})
            solution_bak_data = str({solution_exec:solution_data})
            ##break
            #solution_bak_data = current_bak_data

        if 'neo4j' in solution_type:
            solution_prompt = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，针对以下问题，生成相应的sql指令:\n" + solution_exec
            #import ipdb
            #ipdb.set_trace()
            solution_prompt = solution_prompt.strip()
            solution_output = get_agent(solution_prompt)
            solution_bak_data = zhishiku.zsk[3]['zsk'].find_by_sql(solution_output)
            if solution_bak_data:
                #solution_bak_data = str(solution_bak_data)
                solution_bak_data = str({solution_exec:solution_bak_data})
        if '数据库' in solution_type:
            solution_prompt = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，针对以下问题，生成相应的sql指令:\n" + solution_exec
            #solution_prompt = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，请解决以下问题:\n" + solution_exec
            solution_prompt = solution_prompt.strip()
            solution_output = get_agent(solution_prompt)
            #solution_output = get_agent(solution_prompt)
            #solution_output = "select `企业名称`,`企业类型`,`产业` from `企业数据` where  城市 like '%景德%' limit 10;"
            print(solution_exec + ':' + solution_output)
            #solution_bak_data = zhishiku.zsk[1]['zsk'].find_by_sql(solution_output)
            #import ipdb
            #ipdb.set_trace()
            solution_bak_data = zhishiku.zsk[9]['zsk'].find_by_sql(solution_output)
            if solution_bak_data:
                #solution_bak_data = str(solution_bak_data) 
                solution_bak_data = str({solution_exec:solution_bak_data})
            #solution_data_df = pd.DataFrame(solution_data)
            #if solution_data:
            #    if len(solution_data) == 1 and ('0' in str(solution_data) or 'None' in str(solution_data)):
            #        solution_bak_data = ''
            #        #continue
            #    #is_break = True
            #    if '企业数量' in solution_data_df:
            #        #solution_data = solution_data_df['企业数量'][0]
            #        solution_bak_data = solution_data_df['企业数量'][0]

            #    #solution_data_res.append({solution_exec:solution_data})
            #    #break
        if '使用工具' in solution_type:
            #import ipdb
            #ipdb.set_trace()
            li = solution_exec.split('\t')
            fun,paramater = li[0],li[1:]
            paramater = [str(x)  for x in paramater ]
            if fun == 'python eval':
                #paramater = paramater[0]
                #solution_data = eval(paramater)
                solution_data = eval(solution_exec)
                if solution_data:
                    #solution_data_res.append({solution_exec:solution_data})
                    #break
                    solution_bak_data = str({solution_exec:solution_data})
                    #solution_bak_data = str(solution_data)
            else:
                #paramater = ",".join([f"'{x}'" if isinstance(x,str) else str(x) for x in paramater])
                ##fun,paramater = solution_exec.split('\t')
                #solution_exec = fun + '(' + f'{paramater}' + ')'
                solution_data = chanyeku.chanye(solution_exec)
                if solution_data:
                    #is_break = True
                    #solution_data_res.append({solution_exec:solution_data})
                    #solution_bak_data = str(solution_data)
                    solution_bak_data = str({solution_exec:solution_data})
                    #break
        #if 'document' in solution_type:
        if '搜索文件' in solution_type:
            #import ipdb
            #ipdb.set_trace()
            solution_prompt = solution_exec
            solution_data = zhishiku.zsk[2]['zsk'].find(solution_prompt) #搜索引擎
            if solution_data:
                #solution_bak_data = solution_data
                solution_bak_data = str({solution_exec:solution_data})
        if '搜索引擎' in solution_type:
            solution_prompt = solution_exec
            #solution_data = zhishiku.zsk[1]['zsk'].find(solution_prompt) #mysql 缓存
            #import ipdb
            #ipdb.set_trace()
            #if not solution_data:
            #    solution_data = zhishiku.zsk[2]['zsk'].find(solution_prompt) #文档
                #if len(solution_data) > 0:
                #    zhishiku.zsk[1]['zsk'].save(solution_prompt,solution_prompt,solution_data,'','')
                #    print('save {} mysql successfully'.format(solution_prompt))
            #if not solution_data:
            solution_data = zhishiku.zsk[0]['zsk'].find(solution_prompt) #搜索引擎
                #if len(solution_data) > 0:
                #    zhishiku.zsk[1]['zsk'].save(solution_prompt,solution_prompt,solution_data,'','')
                #    print('save {} mysql successfully'.format(solution_prompt))
            if solution_data:

                print(solution_data)
                #text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
                #is_break = True
                #break
                #solution_data_res.append({solution_exec:solution_data})
                solution_bak_data = str({solution_exec:solution_data})
                #solution_bak_data = solution_data
                #break
        return solution_bak_data
    except Exception  as e:
        return []
def get_solution_data(current_plans,zhishiku,chanyeku):
    solution_data = ''
    #is_break = False
    #solution_data_res = []
    solution_add_data = []
    #import ipdb
    #ipdb.set_trace()
    current_add_data = ''
    #current_plans = [[['从企业数据库中获取数据:烟台新能源企业数量']],[['从企业数据库总获取数据: 烟台新能源企业分布']],[['搜索引擎: 烟台新能源龙头企业']],['llm:烟台新能源企业现状']]
    for current_add_plan in current_plans:
        current_update_data = ''
        if isinstance(current_add_plan,list):
            for current_update_plan in current_add_plan:
                current_bak_data = ''
                if isinstance(current_update_plan,list):
                    for current_bak_plan in current_update_plan:
                        current_bak_data = exec_step(current_bak_plan,zhishiku,chanyeku,current_update_data)
                        if current_bak_data:
                            current_update_data = current_bak_data
                            break
                    if not current_bak_data:
                        print('current_bak_data should be empty,flag1')
                else:
                    current_bak_data = exec_step(current_update_plan,zhishiku,chanyeku,current_update_data)
                    if current_bak_data:
                        current_update_data = current_bak_data
                    else:
                        print('current_bak_data should be empty,flag2')

            current_add_data = current_update_data
        else:
            current_update_data = exec_step(current_update_plan,zhishiku,chanyeku)
            if current_update_data:
                current_add_data = current_update_data
            else:
                print('current_update_data should be empty,flag1')
        if current_add_data:
            solution_add_data.append(current_add_data)
                #solution_update_data = solution_bak_data
            #solution_add_data.append(solution_update_data)

        #if is_break:
        #    break
    #return solution_data
    #return solution_data_res
    return solution_add_data
def build_question_and_context(question,data:pd.DataFrame,format_str='markdown'):
    question_context = ''
    if isinstance(data,list):
        data = pd.DataFrame(data)
    if  data.empty:
        return question
    if format_str == 'markdown':
        question_context = data.to_markdown() + '\n' + question
    return question_context
def get_answer_with_context(prompt,context_data,history_data,instruction):
    #solution_prompt = context_data + '\n\n,从上述文本中，精确的选取有用的部分，回答下面的问题\n' + prompt
    if not isinstance(context_data,str):
        context_data = str(context_data)
    solution_prompt = context_data + ' ' + prompt
    #solution_prompt = context_data + '\n' + '上述文本是和问题相关的文本，请精确的回答下述问题,回答内容中不要出现"根据文本提供的内容"等类似字样:\n' + prompt
    #instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请回答以下问题:"
    solution_prompt = instruction + '\n' + solution_prompt
    #solution_prompt = solution_prompt.strip()
    #import ipdb
    #ipdb.set_trace()
    if history_data:
        history_data.append({"role": "user", "content": solution_prompt})
        solution_prompt = history_data 

    answer = get_stream_llm(solution_prompt)
    #answer = get_stream_llm(history_data)
    return answer

def generate_answer(solution_data,prompt,current_plan,history_data,zhishiku,init_question):
    """
    "{'获取数据': ['从企业数据库中获取数据:获取位于景德镇珠山的企业,>给出企业名称，企业类型,产业', '查询搜索引擎:烟台企业'], '生成答案': [\"'从上述>列表中，选出企业列表'\"], '评价答案': []}"
    {'获取数据': [['从企业数据库中获取数据:北京专精特新企业的企业名称，产业，企业类型'], ['查询搜索引擎:北京专精特新企业']], '生成答案': ['获取答案的前缀', '将答案和前缀进行组合输出'], '评价答案': []}
    """
    #if isinstance(solution_data,list):
    #    solution_data = pd.DataFrame(solution_data)
    prefix = None
    answer = None
    for current in current_plan:
        #if '从上述列表中' in current:
        ##if 'query_llm' in current:
        #    context_question = build_question_and_context(prompt,solution_data)
        #    context_question = context_question.strip()
        #    solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团>队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请结合给定的数据,解决以下问题:\n' + context_question
        #    answer = get_agent(solution_prompt)
        #    #answer = get_agent(solution_prompt)
        #    if not answer:
        #        raise ValueError('没有获得答案，抛出异常，让生成式模型来获取答案')
        if '获取答案的前缀' in current:
        #if '前缀' in current:
            #solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，针对以下问题，生成相应的回答前缀:\n' + prompt
            solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，针对以下问题，生成相应的回答前缀:\n' + prompt
            #import ipdb
            #ipdb.set_trace()
            plan_history_data = get_plan_history(history_data)
            plan_history_data.append({"role":"user","content":solution_prompt})
            #idx = prompt.find('企业')
            #new_prompt = prompt[0:idx+2]
            #solution_prompt = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，请结合给定的数据,解决以下问题:\n' + new_prompt
            #prefix = get_agent(solution_prompt)
            prefix = get_agent(plan_history_data)
            ##prefix = get_agent(solution_prompt)
            #print('prefix:' + prefix)
            ##import ipdb
            ##ipdb.set_trace()
            #if not prefix:
            #    raise ValueError('没有获得答案，抛出异常，让生成式模型来获取答案')
            #answer = get_answer_with_context(prompt,str(solution_data),[])
            #answer = get_answer_with_context(prompt,'\n'.join(solution_data),[])
            #answer = get_answer_with_context(prompt,'\n'.join(solution_data),history_data)
            #break
        if 'llm' in current:
            #import ipdb
            #ipdb.set_trace()
            current_li = current.split(':')
            prompt = current_li[1]
            new_prompt = prompt
            if solution_data:
                new_prompt = str(solution_data) + ' ' + prompt
            answer = get_zhishiku_llm(new_prompt)
        if '将数据和问题进行拼接送给HX LLM模型' in current:
            #import ipdb
            #ipdb.set_trace()
            #answer = get_answer_with_context(prompt,'\n'.join(solution_data),history_data)
            #answer = get_answer_with_context(prompt,'\n'.join(solution_data),[])
            instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请回答以下问题:"
            #solution_prompt = instruction + '\n' + init_question
            #solution_prompt = solution_prompt.strip()
            #answer = get_answer_with_context(init_question,'\n'.join(solution_data),[],instruction)
            answer = get_answer_with_context(init_question,str(solution_data),[],instruction)
            break
        if '合并报告' in current or current=='生成pdf报告':
            #import ipdb
            #ipdb.set_trace()
            file_path = zhishiku.zsk[8]['zsk'].build(solution_data)
            #report_data[key] = sub_report_data
            #report_data = generate_llm_report_data(solution_data)
            #report_data = solution_data
            #report_path = zhishiku.zsk[10]['zsk'].report(report_data)
            net_file_path = f'http://10.0.0.12:17866/download?file_name={file_path}'
            answer = net_file_path
            break
        if '生成pdf报告' in current:
            #import ipdb
            #ipdb.set_trace()
            if ':' in current:
                _,init_question = current.split(':')
            #import ipdb
            #ipdb.set_trace()
            keys = ['标题','摘要','数据','结论']
            report_data = {}
            #context_data = [str(['张家口超高清视频显示产业方向报告', '张家口超高清视频显示产业方向产业报告摘要：张家口超高清视频显示产业方向产业报告旨在全面了解张家口超高清视频显示产业方向的发展情况，包括产业规模、技术水平、市场状况、发展趋势等。报告中还分析了张家口超高清视频显示产业方向面临的挑战和机遇，并提出了相应的发展建议。报告旨在为张家口超高清视频显示产业方向的发展提供参考。', [{'企业数量': 0}], {'张家口超高清视频显示产业企业分布': []}, '张家口超高清视频显示产业方向>报告旨在全面了解张家口超高清视频显示产业方向的发展情况，包括产业规模、技术水平、市场状况、发展趋势等。报告中还分析了张家口超高清视频显示产业方向面临的挑战和机遇，并提出了相应的发展建议。报告旨在为张家口超高清视频显示产业方向的发展提供参考。根据报告，张家口超高清视频显示产业方向的企业数量为0家，企业分布情况未提及。'])]
            #for key in keys:
                #context_data = str(solution_data) + f'针对上述内容，生成报告的{key}'
                #context_data = '\n'.join(solution_data) + f'针对上述内容，生成报告的{key}'
            instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请对以下问题进行回答，并生成产业报告的格式:"
            #solution_prompt = instruction + '\n' + str(solution_data) + ' ' + init_question
            solution_prompt = instruction + '\n' + str(solution_data[0:2]) + ' ' + init_question
            solution_prompt = solution_prompt.strip()
            #answer = get_answer_with_context(solution_prompt,'\n'.join(solution_data),[])
            #context_data = instruction + '\n' + '\n'.join(solution_data) 
            #context_data = instruction + '\n' + '\n'.join(context_data) 
            #sub_report_data = get_llm(context_data)
            sub_report_data = get_zhishiku_llm(solution_prompt)
            #sub_report_data = eval(sub_report_data)
            #file_path = zhishiku.zsk[8]['zsk'].report(sub_report_data)
            sub_report_data_answer = zhishiku.zsk[8]['zsk'].report(sub_report_data)
            ##report_data[key] = sub_report_data
            ##report_data = generate_llm_report_data(solution_data)
            ##report_data = solution_data
            ##report_path = zhishiku.zsk[10]['zsk'].report(report_data)
            #net_file_path = f'http://10.0.0.12:17866/download?file_name={file_path}'
            #answer = net_file_path
            answer = sub_report_data_answer
            break
        if '将数据合并成一个字符串' in current:
            #solution_data = str(solution_data)
            solution_data = '\n'.join(solution_data)
        if '直接作为答案输出' in current:
            if solution_data:
                answer = str(solution_data)
            else:
                answer = get_stream_llm(history_data)
        if '将答案进行组合输出' in current:
            instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请回答以下问题:"
            #solution_prompt = instruction + '\n' + init_question
            #solution_prompt = solution_prompt.strip()
            #answer = get_answer_with_context(init_question,'\n'.join(solution_data),[],instruction)
            answer = get_answer_with_context(init_question,str(solution_data),[],instruction)
            #    #answer = get_answer_with_context('将内容进行精炼，形成一段文本',str(solution_data),[])
            #    #answer = get_answer_with_context(prompt,str(solution_data),[])
            #    answer = get_answer_with_context(prompt,'\n'.join(solution_data),[])
        if '将答案和前缀进行组合输出' in current:
            #import ipdb
            #ipdb.set_trace()
            if not prefix:
                 raise ValueError('没有获得答案，抛出异常，让生成式模型来获取答案')
            else:
                #import ipdb
                #ipdb.set_trace()
                if isinstance(solution_data,list):
                    solution_data = solution_data[0]
                    if isinstance(solution_data,str):
                        try:
                            solution_data = eval(solution_data)
                        except:
                            pass
                    if isinstance(solution_data,dict):
                        solution_data_value = list(solution_data.values())
                        solution_data = solution_data_value[0]
                    try:
                        solution_data = pd.DataFrame(solution_data)
                    except:
                        pass
                if isinstance(solution_data,pd.DataFrame):
                    if len(solution_data) == 1 and len(solution_data.columns) == 1: 
                        answer = prefix + solution_data.to_string(index=False,header=False)
                    else:
                        answer = prefix + ':\n' + solution_data.to_string(index=False,header=False)
                else:
                    if isinstance(solution_data,str):
                        instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请回答以下问题:"
                        #solution_prompt = instruction + '\n' + init_question
                        #solution_prompt = solution_prompt.strip()
                        answer = get_answer_with_context(init_question,str(solution_data),[],instruction)
                        #answer = get_answer_with_context(init_question,'\n'.join(solution_data),[],instruction)
                        #answer = get_answer_with_context(prompt,solution_data,history_data)
                        #answer = get_answer_with_context(prompt,solution_data,[])
        if '直接将问题送给模型' in current:
            history_data.append({"role": "user", "content": prompt})
            answer = get_stream_llm(history_data)
        if '将数据和问题进行拼接送给HX LLM模型' in current:
            instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请回答以下问题:"
            #solution_prompt = instruction + '\n' + init_question
            #solution_prompt = solution_prompt.strip()
            #answer = get_answer_with_context(init_question,'\n'.join(solution_data),[],instruction)
            answer = get_answer_with_context(init_question,str(solution_data),[],instruction)
            #answer = get_answer_with_context(prompt,'\n'.join(solution_data),[])
        if '将答案和模型进行结合' in current or '将答案与模型进行结合' in current:
            instruction = "你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请回答以下问题:"
            #solution_prompt = instruction + '\n' + init_question
            #solution_prompt = solution_prompt.strip()
            answer = get_answer_with_context(init_question,str(solution_data),[],instruction)
            #answer = get_answer_with_context(init_question,str(solution_data),[],instruction)
            #answer = get_answer_with_context(prompt,'\n'.join(solution_data),[])
            #else:
            #    solution_data = get_web_data(prompt,zhishiku)
            #    #answer = get_answer_with_context(prompt,solution_data,history_data)
            #    answer = get_answer_with_context(prompt,solution_data,[])
    #else:
    #else:
    #    for token in answer.split('\n'):
    #        current_content += token + '\n'
    #        time.sleep(0.005)
    #        yield current_content
    return answer
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
#def transform_openai2llama2(history_formatted):
#    FIRST_PROMPT_TEMPLATE = (
#            "<s>[INST] <<SYS>>\n"
#            "You are a helpful assistant. 你是一个乐于助人的助手。\n"
#            "<</SYS>>\n\n{instruction} [/INST] {response} </s>"
#        )
#    SECOND_PROMPT_TEMPLATE = (
#            "<s>[INST] {instruction} [/INST] {response} </s>"
#        )
#    res = []
#    prompt = ''
#    is_editored = False
#    if history_formatted is not None:
#        for i, old_chat in enumerate(history_formatted):
#            #if old_chat['role'] == "system":
#            #    sys_prompt = (
#            #" <<SYS>>\n"
#            #"You are a helpful assistant. 你是一个乐于助人的助手。\n"
#            #"<</SYS>>\n\n"
#            #    )
#            #    res.append(sys_prompt)
#            if i%2 == 0:
#                if i/2 == 0:
#                    prompt = FIRST_PROMPT_TEMPLATE
#                else:
#                    prompt = SECOND_PROMPT_TEMPLATE
#                is_editored = False
#
#            if old_chat['role'] == "user":
#                #history_data.append(
#                #    {"role": "user", "content": old_chat['content']},)
#                #user_prompt = (
#                prompt=prompt.replace('{instruction}',old_chat['content'])
#                is_editored = True
#            elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
#                #history_data.append(
#                #    {"role": "assistant", "content": old_chat['content']},)
#                #prompt.format({'response':response})
#                prompt=prompt.replace('{response}',old_chat['content'])
#                is_editored = True
#            if i%2 == 1 and is_editored:
#                res.append(prompt)
#            if i == len(history_formatted) - 1 and is_editored:
#                prompt=prompt.replace('{response}','')
#                res.append(prompt)
#    return ''.join(res)
def get_step_output(output,zhishiku,chanyeku,prompt,history_data,return_stream=True):
    steps = ['获取数据','生成答案','评价答案']
    #output_type = output["type"]
    #output = output['content']
    answer = None
    if output["type"] == "llm":
        answer = get_stream_llm(prompt)
    if output["type"] == "answer":
        answer = output["content"]
    if output["type"] == "step" or output["type"]== "tools" or output["type"]== "plan":
        output = output['content']
        for step in steps: 
            current_plan = output[step]
            if step == '获取数据':
                solution_data = get_solution_data(current_plan,zhishiku,chanyeku)
            if step == '生成答案':
                answer = generate_answer(solution_data,prompt,current_plan,history_data,zhishiku)
    if answer is None:
        raise ValueError('answer为None，抛出异常')
    if isinstance(answer,str):
        #current_content = ''
        #for token in answer.split('\n'):
        #    current_content += token + '\n'
        #    time.sleep(0.05)
        #    #yield current_content
        #    yield current_content.replace('\n','<br />\n')
        return answer
    else:
        if return_stream:
            return answer
        else:
            current_str = ''
            for chunk in answer:
                if '[DONE]' in chunk:
                    continue
                if len(chunk) > 2:
                    chunk = json.loads(chunk)
                    #yield chunk["response"].replace('\n','<br />\n')
                    #current_str += chunk["response"] 
                    current_str = chunk["response"] 
            #import ipdb
            #ipdb.set_trace()
            return current_str
def decomposer_plan(output,prompt,history_data,zhishiku,chanyeku,init_question):
    output_type = output["type"]
    output = output['content']
    solution_datas = []
    plans = output['获取数据']
    current_plan = plans[2:3]
    #import ipdb
    #ipdb.set_trace()
    for plan in current_plan:
        if not isinstance(plan,str):
            if isinstance(plan,list):
                np_plan = np.array(plan)
                plan = np_plan.squeeze().tolist() 
        plan_question = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，针对以下问题，生成相应的解决问题的计划与步骤:\n' + plan
        current_output = get_agent(plan_question)
        """
{'type': 'step', 'content': {'获取数据': [[['从企业数据库中获取数据:烟台新能源企业数量']], [
['从企业数据库总获取数据: 烟台新能源企业分布']], [['搜索引擎: 烟台新能源龙头企业']]], '生成
答案': ['llm:烟台新能源企业现状'], '评价答案': []}, 'init_question': '烟台新能源企业现状,包
含烟台新能源现有企业规模,分布,以及龙头企业等'}
        """
        #import ipdb
        #ipdb.set_trace()
        current_output = eval(current_output)
        #current_output['content']['生成答案']=['生成pdf报告'] 
        plan_data = execution(current_output,prompt,history_data,zhishiku,chanyeku,init_question)
        #solution_data = get_step_output(current_output,zhishiku,chanyeku,prompt,history_data,return_stream=False)
        solution_datas.append(plan_data)
    solution_data = solution_datas
    plan = output['生成答案']
    answer = generate_answer(solution_data,prompt,plan,history_data,zhishiku,init_question)
    #return solution_data
    return answer

def execution(output,prompt,history_data,zhishiku,chanyeku,init_question):
    """
    数据可以获取多轮，但是生成的结果只有一个;
    一个完整的execution需要包含获取数据和生成答案选项，如果直接需要获取数据，生成答案选项可以写直接输出；
    如果需要对不同的数据生成不同的结果，然后再进行组合的话，需要放到plan里面。
    """
    steps = ['获取数据','生成答案','评价答案']
    output_type = output["type"]
    output = output['content']
    answer = ''
    for step in steps: 
        current_plan = output[step]
        if step == '获取数据':
            solution_data = get_solution_data(current_plan,zhishiku,chanyeku)
            if not output['生成答案']:
                return solution_data
        if step == '生成答案':
            answer = generate_answer(solution_data,prompt,current_plan,history_data,zhishiku,init_question)
            #if answer is None:
            #    raise ValueError('answer为None，抛出异常')
            #if isinstance(answer,str):
            #    current_content = ''
            #    for token in answer.split('\n'):
            #        current_content += token + '\n'
            #        time.sleep(0.05)
            #        #yield current_content
            #        yield current_content.replace('\n','<br />\n')
            #else:
            #    for chunk in answer:
            #        #print(chunk)
            #        #if chunk['choices'][0]["finish_reason"]!="stop":
            #        #    if hasattr(chunk['choices'][0]['delta'], 'content'):
            #        #        resTemp+=chunk['choices'][0]['delta']['content']
            #        #        yield resTemp
            #        if '[DONE]' in chunk:
            #            continue
            #        if len(chunk) > 2:
            #            chunk = json.loads(chunk)
            #            yield chunk["response"].replace('\n','<br />\n')
            return answer
def get_plan_history(history_data):
    res = []
    for da in history_data:
        if da['role'] == 'system' or da['role'] == 'user':
            res.append(da)
    return res
#def chat_one(prompt, history_formatted, max_length, top_p, temperature, data):
def chat_one(prompt, history_formatted, max_length, top_p, temperature, web_receive_data,zhishiku=False,chanyeku=False):
    history_data = [ {"role": "system", "content": "You are a helpful assistant. 你是一个乐于助人的助手。\n"}]
    # web 界面history正常，G端history最新的问题在最后，需要调整顺序，小程序需要修改格式。这些都放在wenda程序里修改吧。
    #history_formatted = history_formatted[::-1]
    #history_formatted = history_formatted[-5:]
    #if history_formatted is not None:
    #    for i, old_chat in enumerate(history_formatted):
    #        if 'role' in old_chat:
    #            if old_chat['role'] == "user":
    #                history_data.append(
    #                    {"role": "user", "content": old_chat['content']})
    #            elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
    #                if i > len(history_formatted) - 4:
    #                    history_data.append(
    #                        {"role": "assistant", "content": old_chat['content']},)
    #        else:
    #            history_data.append({"role":"user","content":old_chat["question"]})
    #            history_data.append({"role":"assistant","content":old_chat["answer"]})
    #history_data.append({"role": "user", "content": prompt})
    #import ipdb
    #ipdb.set_trace()
    history_data = history_formatted

    #import ipdb
    #ipdb.set_trace()
    #history_data = transform_openai2llama2(history_data)
    prompt = prompt.strip()
    is_file = False
    if '学习已经完成' in str(history_formatted):
        is_file = True
    plan_question = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，针对以下问题，生成相应的解决问题的计划与步骤:\n' + prompt
    #plan_question = plan_question.strip()
    #plan_history_data = copy.deepcopy(history_data)
    plan_history_data = get_plan_history(history_data)
    #import ipdb
    #ipdb.set_trace()
    plan_history_data.append({"role":"user","content":plan_question})
    #plan_history_data = plan_history_data[::-1]
    print(history_data)
    content = ''.join([x['content'] for x in plan_history_data])

    if len(content) > 7000:
        #import ipdb
        #ipdb.set_trace()
        history_data = []
        history_data.append({"role": "user", "content": prompt},)
        if len(prompt) > 8000:
            raise ValueError('最长只能支持8000个字符，不要超标')
            #history_data.extend(plan_history_data[-20:])
    #output = get_agent(plan_question)
    #import ipdb
    #ipdb.set_trace()
    output = get_agent(plan_history_data)
    print(output)
    #output = get_agent(plan_history_data)
    solution_data = ''
    #import ipdb
    #ipdb.set_trace()
    final_answer = ''
    is_normal = 1
    try:
        """
"{'获取数据': ['从企业数据库中获取数据:获取位于景德镇珠山的企业,>给出企业名称，企业类型,产业', '查询搜索引擎:烟台企业'], '生成答案': [\"'从上述>列表中，选出企业列表'\"], '评价答案': []}"
    {'获取数据': [['从企业数据库中获取数据:北京专精特新企业的企业名称，产业，企业类型'], ['查询搜索引擎:北京专精特新企业']], '生成答案': ['获取答案的前缀', '将答案和前缀进行组合输出'], '评价答案': []}
        """
        #xy = 2/0 
        if is_file:
            output = str({'type': 'step', 'content': {'获取数据': [[['搜索文件:{}'.format(prompt), f'查询搜索引擎:{prompt}']]], '生成答案': ['将答案与模型进行结合'], '评价答案': []},'init_question':prompt})
        output = eval(output)
        init_question = output['init_question']
        #output['获取数据']=[['搜索引擎:{}'.format(prompt)]]
        solution_data = ''
        print(output)
        #import ipdb
        #ipdb.set_trace()
        steps = ['获取数据','生成答案','评价答案']
        solution_data = ''
        if "type" not in output:
            output = {'type':'step','content':output}
        if output["type"] == "llm":
            """
            对于一些娱乐问题或者与产业不是太相关问题，直接让llm来作答
            """
            response = get_stream_llm(prompt)
            #resTemp=""
            #for chunk in response:
            #    if '[DONE]' in chunk:
            #        continue
            #    if len(chunk) > 2:
            #        chunk = json.loads(chunk)
            #        yield chunk["response"].replace('\n','<br />\n')
            answer = response
        if output["type"] == "answer":
            """
            比如问功能，问你是谁的问题，agent直接出答案
            """
            response = output["content"]
            answer = response
            #curr = ''
            #for chunk in list(response):
            #    curr += chunk
            #    time.sleep(0.05)
            #    yield curr.replace('\n','<br />\n')
        if output["type"] == "plan":
        #    output_type = output["type"]
        #    output = output['content']
        #if output["type"] == "step" or output["type"]== "tools" or output["type"]== "plan":
            
            answer = decomposer_plan(output,prompt,history_data,zhishiku,chanyeku,init_question)
        if output["type"] == "step" or output["type"]== "tools":
            answer = execution(output,prompt,history_data,zhishiku,chanyeku,init_question)
        if answer is None:
            raise ValueError('answer为None，抛出异常')
        if isinstance(answer,str):
            final_answer = answer
            current_content = ''
            for token in answer.split('\n'):
                current_content += token + '\n'
                time.sleep(0.05)
                #yield current_content
                yield current_content.replace('\n','<br />\n')
        else:
            for chunk in answer:
                #print(chunk)
                #if chunk['choices'][0]["finish_reason"]!="stop":
                #    if hasattr(chunk['choices'][0]['delta'], 'content'):
                #        resTemp+=chunk['choices'][0]['delta']['content']
                #        yield resTemp
                if '[DONE]' in chunk:
                    continue
                if len(chunk) > 2:
                    chunk = json.loads(chunk)
                    final_answer = chunk["response"]
                    yield chunk["response"].replace('\n','<br />\n')
    except Exception as e:
        is_normal = 0
        print(e)
        #response = completion_with_backoff(model="gpt-4-0613", messages=history_data, max_tokens=2048, stream=True, headers={"x-api2d-no-cache": "1"},timeout=3)
        #response = get_stream_llm(history_data)
        #import ipdb
        #ipdb.set_trace()
        if not solution_data:
            solution_data = get_web_data(prompt,zhishiku)
        instruction = ''
        response = get_answer_with_context(prompt,solution_data,history_data,instruction)
        #response = get_answer_with_context(prompt,solution_data,history_data)
        #import ipdb
        #ipdb.set_trace()
        #response = completion_with_backoff(kwargs)
        resTemp=""
        #import ipdb
        #ipdb.set_trace()
        for chunk in response:
            #print(chunk)
            #if chunk['choices'][0]["finish_reason"]!="stop":
            #    if hasattr(chunk['choices'][0]['delta'], 'content'):
            #        resTemp+=chunk['choices'][0]['delta']['content']
            #        yield resTemp
            if '[DONE]' in chunk:
                continue
            if len(chunk) > 2:
                #import ipdb
                #ipdb.set_trace()
                chunk = json.loads(chunk)
                final_answer = chunk["response"]
                yield chunk["response"].replace('\n','<br />\n')
        #except:
        #    import ipdb
        #    ipdb.set_trace()
        #    print(1)
    zhishiku.zsk[1]['zsk'].save(prompt,str(output),solution_data,final_answer,is_normal)


chatCompletion = None


def load_model():
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    #openai.api_key = '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'
    #openai.api_key = 'sk-YR2Mtp2ht8u0ruHQ1058B5996dFc40C190B22774D5Bc7964'#测试用
    openai.api_key = 'sk-cRujJbZqefFoj5753c8d94B8F7654c57807cCc3b145aC547'
    #openai.api_key = 'fk217408-4KdxNeEDSjmll43jQ0ItKVKmjhkvi7xH'
    openai.api_base = settings.llm.api_host

class Lock:
    def __init__(self):
        pass

    def get_waiting_threads(self):
        return 0

    def __enter__(self): 
        pass

    def __exit__(self, exc_type, exc_val, exc_tb): 
        pass
