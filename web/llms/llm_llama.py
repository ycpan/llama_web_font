from plugins.common import settings

from threading import Thread
from queue import Queue, Empty
from threading import Thread
from collections.abc import Generator
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from .llm_remote import get_output
from .llm_remote import get_output_v1
from typing import Any
import re
import time
import pandas as pd
from .langchain_qa import mymodel
write_sql = False
#n_gpu_layers = 22  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#n_batch = 8096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#n_ctx=8096
n_ctx=5000

#template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
#{question}
#"""
template = """[INST] <<SYS>>
You are a helpful assistant. 你是一个乐于助人的助手。
<</SYS>>
{question}
[/INST]
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
#zhishiku_template = (
#    "Below is an instruction that describes a task. "
#    "Write a response that appropriately completes the request.\n\n"
#    "### Instruction:\n"
#    "以下为背景知识：\n"
#    "{context}"
#    "\n"
#    "请根据以上背景知识, 回答这个问题：{question}。\n\n"
#    "### Response: "
#)
#zhishiku_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
#### Instruction:
#你是一个产业专家，请你提供专业、有逻辑、内容真实、有价值的详细回复。请回答{question}。下面的文本可能会对问题形成干扰，请判断下述文本对回答是否有帮助，如果有帮助请结合下述内容进行回答问题；如果没有帮助，请忽略。
#{context}
#### Response: 
#"""
#You are a helpful assistant. 你是一个乐于助人的助手。
zhishiku_template = """[INST] <<SYS>>
You are a helpful assistant. 你是一个乐于助人的助手。请你提供>专业、有逻辑、内容真实、有价值的详细回复。
<</SYS>>


请回答{question}。回答内容中不要出现"根据文本提供的内容"等类似字样。
{context}
 [/INST]
"""
#你是一个产业专家，请回答{question}。下面的文本可能会对问题形成干扰，请判断下述文本对回答是否有帮助，如果有帮助请结合下述内容进行回答问题；如果没有帮助，请忽略。
#请选择性的使用上述文本，结合和问题相关的内容，请回答{question},尽量减少重复表达。
zhishiku_prompt = PromptTemplate(template=zhishiku_template, input_variables=["question","context"])
# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
#class QueueCallback(BaseCallbackHandler):
#    """Callback handler for streaming LLM responses to a queue."""
#
#    def __init__(self, q):
#        self.q = q
#
#    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#        self.q.put(token)
#
#    def on_llm_end(self, *args, **kwargs: Any) -> None:
#        return self.q.empty()
#
#
## Create a function that will return our generator
#def mystream(input_text) -> Generator:
#
#    # Create a Queue
#    q = Queue()
#    job_done = object()
#
#    # Initialize the LLM we'll be using
#    #llm = OpenAI(
#    #    streaming=True, 
#    #    callbacks=[QueueCallback(q)], 
#    #    temperature=0
#    #)
#    llm = LlamaCpp(
#        model_path=settings.llm.path, callbacks=[QueueCallback(q)], verbose=True, n_gpu_layers=settings.llm.n_gpu_layers,n_ctx=n_ctx, n_batch=n_batch,max_tokens=n_ctx,temperature=0.2,client='Alpaca'
#        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/7B/ggml-model-q4_0.bin", callbacks=[QueueCallback(q)], verbose=True, n_gpu_layers=n_gpu_layers,n_ctx=n_ctx, n_batch=n_batch,max_tokens=1024,temperature=0.2,client='Alpaca'
#        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/33B/ggml-model-q4_0.bin", callbacks=[QueueCallback(q)], verbose=True, n_gpu_layers=n_gpu_layers, n_batch=n_batch,max_tokens=1024,temperature=0.2,client='Alpaca'
#        )
#
#    # Create a funciton to call - this will run in a thread
#    def task():
#        resp = llm(input_text)
#        q.put(job_done)
#
#    # Create a thread and start the function
#    t = Thread(target=task)
#    t.start()
#
#    content = ""
#
#    # Get each new token from the queue and yield for our generator
#    while True:
#        try:
#            next_token = q.get(True, timeout=1)
#            if next_token is job_done:
#                break
#            content += next_token
#            yield next_token, content
#        except Empty:
#            continue
if settings.llm.strategy.startswith("Q"):
    runtime = "cpp"

    def chat_init(history):
        history_formatted = None
        if history is not None:
            history_formatted = ""
            for i, old_chat in enumerate(history):
                if old_chat['role'] == "user":
                    history_formatted+="Q: "+old_chat['content']+'\n'
                elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                    history_formatted+=" A: "+old_chat['content']+'\n'
                else:
                    continue
        return history_formatted+" "


    def chat_one(question, history_formatted, max_length, top_p, temperature,data, zhishiku=False,chanyeku=False):
        def generate_prompt(instruction):
            #return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
        ### Instruction:
        #{instruction}
        
        ### Response: """
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        {instruction}
        """
        #import ipdb
        #ipdb.set_trace()
        #if zhishiku:
        #    prompt=history_formatted+"%s\nAssistant: "%prompt
        #else:
        #    #prompt=history_formatted+"Human: %s\nAssistant: "%prompt
        #    prompt=history_formatted+generate_prompt(prompt)
        #stream = model(prompt,
        #stop=["Human:","### Hum","### Instruction:"], temperature=temperature,max_tokens=max_length, top_p=top_p,stream=True,model='Alpaca')
        #stop=["Human:","### Hum","### Instruction:"], temperature=0.2,max_tokens=1024, stream=True,model='Alpaca')
        # print(output['choices'])
        #text=""
        #for output in stream:
        #    text+=output["choices"][0]["text"]
        #    yield text
        #question = "北京市的产业特点"
        #prompt = PromptTemplate(template=template, input_variables=["question"])
        #import ipdb
        #ipdb.set_trace()
        myprompt = prompt.format(question=question)
        #output = model.llm.predict(myprompt)
        #plan_question = '你是一个问题解决专家，请生成相应的计划与步骤\n' + question
        #plan_question = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题解决专家，请解决以下问题:\n' + question
        plan_question = '你的名字叫小星，一个产业算法智能助手，由合享智星算法团队于2022年8月开发，可以解决产业洞察，诊断，企业推荐等相关问题。现在，你作为产业问题>解决专家，请解决以下问题:\n' + question
        plan_question = plan_question.strip()
        output = get_output(plan_question)
        #output = get_output_v1(question)
        print('规划\n' + output)
        #pattern = r"Step(\d+):\s+(.*?)\n``(.*?)``"
        zhishiku_content = ''
        #import ipdb
        #ipdb.set_trace()
        answer = ''
        if '专利' in question:
            answer=output
        #else 'select' in output:
        #else:
        elif 'Step' in output:
            #pattern = r"Step(\d+):(.*?)\n(.*?):``(.*?)``"
            pattern = r'Step(\d+):(.*?):\n``(.*?)``'
            matches = re.findall(pattern, output, re.DOTALL)
            #for step,thought,exec_type,exec_content in matches:
            for step,exec_type,exec_content in matches:
                if exec_type == 'exec queryDB':
                    try:
                        content = zhishiku.zsk[1]['zsk'].find_by_sql(exec_content)
                    except Exception as e:
                        content = []
                        print('捕获到异常:{}'.format(e))
                    answer = pd.DataFrame(content)
                    print('数据库获取')
                    print(answer)

                    #if len(answer) == 0:
                    #    answer = '我是一个AI模型，我没有这方面的详细数据，不能回答你的问题'
                        #myprompt = zhishiku_prompt.format(question=question,context=content)
                        #mystream = model.stream(myprompt)
                        #for next_token, content in mystream:
                        #    yield content
                    if len(answer) > 0:
                        #answer = answer.to_markdown()
                        #answer = answer.to_string(index=False)
                        answer.index = answer.index + 1
                        answer = answer.to_string(index=True,header=False)
                        answer = answer.replace('\n','<br />\n')
                        break
                    else:
                        answer = ''
                    #current_content = ''
                    #for token in answer.split('\n'):
                    #    current_content += token
                    #    time.sleep(0.005)
                    #    yield current_content
                    #zhishiku_content = answer
                if exec_type == 'exec queryApp':
                    #import ipdb
                    #ipdb.set_trace()
                    answer = ''
                    try:
                        data = chanyeku.chanye(exec_content)
                        answer = data
                        answer = answer.replace('\n','<br />\n')
                        break
                    except Exception as e:
                        print(e)
                if exec_type == 'exec queryWEB':
                    zhishiku_context = zhishiku.zsk[1]['zsk'].find(exec_content)
                    #import ipdb
                    #ipdb.set_trace()
                    #zhishiku_context = zhishiku.zsk[1]['zsk'].find(question)
                    if not zhishiku_context:
                        zhishiku_context = zhishiku.zsk[0]['zsk'].find(exec_content)
                        if len(zhishiku_context) > 0:
                            zhishiku.zsk[1]['zsk'].save(exec_content,exec_content,zhishiku_context,'','')
                            print('save mysql successfully')
                    if len(zhishiku_context) > 0:
                        myprompt = zhishiku_prompt.format(question=exec_content,context=zhishiku_context)
                        #mystream = model.stream(myprompt)
                        #for next_token, content in mystream:
                        #    yield content
                        break
                if exec_type == 'extract db content':
                    new_question = exec_content
                    if len(zhishiku_content.split('\n')) < 2:
                        continue
                    else:
                        new_qu = zhishiku_content[0:3500] + '\n' + new_question + '，在答案中不要出现从上表可以看出字样'
                        #new_qu = zhishiku_content[0:3500] + '\n' + '上表是{},请将上述表格复制输出，然后回答{}'.format(matches[2][2],question) + '，在答案中不要出现从上表可以看出字样'
                        myprompt = prompt.format(question=new_qu)
                        #myprompt = zhishiku_prompt.format(question=question,context=zhishiku_content)
                        #model.llm1.stream(myprompt)
                        
                        #for content in model.llm1.stream(myprompt):
                        #for next_token, content in model.stream1(myprompt):
                        #    #current_content += content
                        #    yield content
                        break
        elif 'llm' not in output:
            answer = output
        #else:
        #myprompt = prompt.format(question=question)
        #current_content = ''
        #for content in model.llm1.stream(myprompt):
        #    current_content += content
        #    yield current_content
        if answer.strip():
            current_content = ''
            for token in answer.split('\n'):
                current_content += token + '\n'
                time.sleep(0.005)
                yield current_content
        else:
            print('prompt\n' + myprompt)


            mystream = model.stream(myprompt)
            #for next_token, content in model.stream(myprompt):
            for next_token, content in mystream:
                #current_content += content
                #yield content
                yield content.replace('\n','<br />\n'),
            #break
            #mystream = model.llm1.stream(myprompt)
            #for next_token, content in mystream:
            #    yield content
            #break
        #if 'select'  in output:
        #    res = zhishiku.zsk[1]['zsk'].find_by_sql(output)
        #    res = str(res)
        #    curr_content = ''
        #    for  content in res:
        #        #print(next_token)
        #        #print(content)
        #        curr_content += content
        #        yield curr_content
        #    #return 

        #else:
        #    curr_content = ''
        #    for  content in output:
        #        #print(next_token)
        #        #print(content)
        #        curr_content += content
        #        yield curr_content
        #    #return 
        #mystream = None
        ##if zhishiku:
        ##    try:
        ##        print('尝试使用mysql加载')
        ##        zhishiku_context = zhishiku.zsk[1]['zsk'].find(question)
        ##        #zhishiku_context = zhishiku.zsk[7]['zsk'].find(question)
        ##        if not zhishiku_context:
        ##            print('mysql为空，准备加载qdrant')
        ##            zhishiku_context = zhishiku.zsk[7]['zsk'].find(question)
        ##            #zhishiku_context = zhishiku.zsk[0]['zsk'].find(question)
        ##            if not zhishiku_context:
        ##                print('mysql为空，准备加载网页')
        ##                zhishiku_context = zhishiku.zsk[0]['zsk'].find(question)
        ##                #write_sql = True
        ##                zhishiku.zsk[1]['zsk'].save(question,question,zhishiku_context,'','')
        ##                print('save mysql successfully')
        ##        print(zhishiku_context)
        ##        if zhishiku_context:
        ##            #mystream = model.qa_stream(zhishiku_context,question)
        ##            myprompt = zhishiku_prompt.format(question=question,context=zhishiku_context)
        ##            mystream = model.stream(myprompt)
        ##    except Exception as e:
        ##        print('in zhishiku process,happend {}'.format(e))
        #if mystream is None:
        #    mystream = model.stream(myprompt)
        ##for next_token, content in mystream(myprompt):
        #for next_token, content in mystream:
        #    #print(next_token)
        #    #print(content)
        #    yield content
        ##import ipdb
        ##ipdb.set_trace()
        ##if write_sql:
        ##    zhishiku.zsk[1]['zsk'].save(question,question,zhishiku_context,content,'')

    def load_model():
        global model
        from llama_cpp import Llama
        model = mymodel()
        
        #try:
        #    #import ipdb
        #    #ipdb.set_trace()
        #    cpu_count = int(settings.llm.strategy.split('->')[1])
        #    model = Llama(model_path=settings.llm.path,use_mlock=True,n_ctx=4096,n_threads=cpu_count)
        #except:
        #    model = Llama(model_path=settings.llm.path,use_mlock=True,n_ctx=4096)

else:
    runtime = "torch"

    user = "Human"
    answer = "Assistant"
    interface = ":"

    import torch
    import gc
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    def chat_init(history):
        tmp = []
        # print(history)
        for i, old_chat in enumerate(history):
            if old_chat['role'] == "user":
                tmp.append(f"{user}{interface} "+old_chat['content'])
            elif old_chat['role'] == "AI":
                tmp.append(f"{answer}{interface} "+old_chat['content'])
            else:
                continue
        history='\n\n'.join(tmp)
        return history
    
    def partial_stop(output, stop_str):
        for i in range(0, min(len(output), len(stop_str))):
            if stop_str.startswith(output[-i:]):
                return True
        return False
    
    def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
    ) -> LogitsProcessorList:
        processor_list = LogitsProcessorList()
        # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list

    @torch.inference_mode()
    def generate_stream(
        model, tokenizer, query: str, max_length=2048, do_sample=True, top_p=1.0, temperature=1.0, logits_processor=None
            ):
        prompt = query
        len_prompt = len(prompt)
        temperature = temperature
        repetition_penalty = 1.0
        top_p = top_p
        top_k = -1  # -1 means disable
        max_new_tokens = 256
        stop_str = '\n\n\n'
        echo = False
        stop_token_ids =  []
        stop_token_ids.append(tokenizer.eos_token_id)
        device = 'cuda'
        stream_interval = 2
        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )

        input_ids = tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)


        max_src_len = max_length - max_new_tokens - 8

        input_ids = input_ids[-max_src_len:]


        past_key_values = out = None
        for i in range(max_new_tokens):
            if i == 0:
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = model.lm_head(out[0])
                else:
                    out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                # prevent yielding partial stop sequence
                if not partially_stopped:
                    yield {
                        "text": output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": None,
                    }

            if stopped:
                break

        # finish stream event, which contains finish reason
        if i == max_new_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }

        # clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()

    def chat_one(prompt, history_formatted, max_length, top_p, temperature, zhishiku=False):
        def generate_prompt(instruction):
            #return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
        ### Instruction:
        #{instruction}
        
        ### Response: """
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        {instruction}"""
        import ipdb
        ipdb.set_trace()
        if prompt.startswith("raw!"):
            print("LLAMA raw mode!")
            ctx=prompt.replace("raw!","")
        else:
            #ctx = f"\n\n{user}{interface} {prompt}\n\n{answer}{interface}"
            ctx = generate_prompt(prompt)
            ctx=history_formatted+ctx
            ctx = ctx.strip('\n')
        yield str(len(ctx))+'字正在计算'
        for response in generate_stream(model,tokenizer, ctx,
                                                max_length=max_length, top_p=top_p, temperature=temperature):
            yield response['text']

    def sum_values(dict):
        total = 0
        for value in dict.values():
            total += value
        return total

    def dict_to_list(d):
        l = []
        for k, v in d.items():
            l.extend([k] * v)
        return l

    def load_model():
        global model, tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        num_trans_layers = 28
        strategy = ('->'.join([x.strip() for x in settings.llm.strategy.split('->')])).replace('->', ' -> ')
        s = [x.strip().split(' ') for x in strategy.split('->')]
        print(s)
        if len(s)>1:
            from accelerate import dispatch_model
            start_device = int(s[0][0].split(':')[1])
            device_map = {'transformer.word_embeddings': start_device,
                    'transformer.final_layernorm': start_device, 'lm_head': start_device}
            
            n = {}
            for i in range(len(s)):
                si = s[i]
                if len(s[i]) > 2:
                    ss = si[2]
                    if ss.startswith('*'):
                            n[int(si[0].split(':')[1])]=int(ss[1:])
                else:
                    n[int(si[0].split(':')[1])] = num_trans_layers+2-sum_values(n)
            n[start_device] -= 2
            n = dict_to_list(n)
            for i in range(num_trans_layers):
                device_map[f'transformer.layers.{i}'] = n[i]

        tokenizer = AutoTokenizer.from_pretrained(
            settings.llm.path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm.path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        import ipdb
        ipdb.set_trace()
        if not (settings.llm.lora == '' or settings.llm.lora == None):
            print('Lora模型地址', settings.llm.lora)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, settings.llm.lora,adapter_name=settings.llm.lora)
            tokenizer = AutoTokenizer.from_pretrained(
                settings.llm.lora, use_fast=False)
            
        device, precision = s[0][0], s[0][1]
        # 根据设备执行不同的操作
        if device == 'cpu':
            # 如果是cpu，不做任何操作
            pass
        elif device == 'cuda':
            # 如果是gpu，把模型移动到显卡
            import torch
            if not (precision.startswith('fp16i') and torch.cuda.get_device_properties(0).total_memory < 1.4e+10):
                model = model.cuda()
        elif len(s)>1 and device.startswith('cuda:'):
            pass
        else:
            # 如果是其他设备，报错并退出程序
            print('Error: 不受支持的设备')
            exit()
        # 根据精度执行不同的操作
        if precision == 'fp16':
            # 如果是fp16，把模型转化为半精度
            model = model.half()
        elif precision == 'fp32':
            # 如果是fp32，把模型转化为全精度
            model = model.float()
        elif precision.startswith('fp16i'):
            # 如果是fp16i开头，把模型转化为指定的精度
            # 从字符串中提取精度的数字部分
            bits = int(precision[5:])
            # 调用quantize方法，传入精度参数
            model = model.quantize(bits)
            if device == 'cuda':
                model = model.cuda()
            model = model.half()
        elif precision.startswith('fp32i'):
            # 如果是fp32i开头，把模型转化为指定的精度
            # 从字符串中提取精度的数字部分
            bits = int(precision[5:])
            # 调用quantize方法，传入精度参数
            model = model.quantize(bits)
            if device == 'cuda':
                model = model.cuda()
            model = model.float()
        else:
            # 如果是其他精度，报错并退出程序
            print('Error: 不受支持的精度')
            exit()
        if len(s)>1:
            model = dispatch_model(model, device_map=device_map)
        model = model.eval()

