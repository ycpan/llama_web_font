import argparse
import os
#file_path = "7.6专精特新企业简介添加企业名称文本.txt"
file_path = "robotic.txt"
#file_path = "烟台产业.txt"
#embedding_path = './all-mpnet-base-v2'
embedding_path = '/devdata/home/user/panyongcan/Project/embedding/m3e-base'
#model_path = args.model_path

import torch
from langchain import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
from collections.abc import Generator
from queue import Queue, Empty
from threading import Thread

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()



simple_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
{question}
"""
simple_prompt = PromptTemplate(template=simple_template, input_variables=["question"])
prompt_template = ("Below is an instruction that describes a task. "
                   "Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\n{context}\n{question}\n\n### Response: ")


refine_prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "这是原始问题: {question}\n"
    "已有的回答: {existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
    "\n\n"
    "{context_str}\n"
    "\\nn"
    "请根据新的文段，进一步完善你的回答。\n\n"
    "### Response: "
)

initial_qa_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "请根据以上背景知识, 回答这个问题：{question}。\n\n"
    "### Response: "
)


#if __name__ == '__main__':
    #load_type = torch.float16
    #if torch.cuda.is_available():
    #    device = torch.device(0)
    #else:
    #    device = torch.device('cpu')
    
class mymodel:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100)

        print("Loading the embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
        print("loading LLM...")
        self.q = Queue()
        self.job_done = object()
        n_gpu_layers = 61  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 512# Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        #n_ctx=5000
        n_ctx=8096
        self.llm = LlamaCpp(
           # model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/33B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers,n_ctx=2048, n_batch=n_batch,max_tokens=1024,temperature=0.3,client='Alpaca',repeat_penalty=1.2,top_k=50,top_p=0.95#n_threads=40
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/7B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,n_batch=n_batch,max_tokens=1524,temperature=0.2,client='Alpaca'
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/33B/ggml-model-q2_k.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,n_batch=n_batch,max_tokens=1024,temperature=0.2,client='Alpaca'
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/33B/ggml-model-q3_k_s.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,n_batch=n_batch,max_tokens=1024,temperature=0.2,client='Alpaca'
        ##model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/33B/ggml-model-q3_k_m.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,n_batch=n_batch,max_tokens=1024,temperature=0.2,client='Alpaca'
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/zh-models/33B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,n_batch=n_batch,max_tokens=1024,temperature=0.2,client='Alpaca'
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/llama2/7B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=4096,temperature=0.75,top_p=1,max_length=2000,eps=1e-5
        model_path="/devdata/home/user/panyongcan/Project/llama.cpp/llama2/13B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=4096,temperature=0.5,top_k=40,top_p=0.9,max_tokens=2000,max_length=2000,suffix='[INST]',eps=1e-5
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/llama2/merge_7B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=4096,temperature=0.5,top_k=40,top_p=0.9,max_tokens=2000,max_length=2000,suffix='[INST]',eps=1e-5
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/llama2/merge_7B_v3/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=4096,temperature=0.5,top_k=40,top_p=0.9,max_tokens=2000,max_length=2000,suffix='[INST]',eps=1e-5
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/llama2/merge_7B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=4096,temperature=0.5,top_k=40,top_p=0.9,max_tokens=2000,max_length=2000,suffix='[INST]',eps=1e-5
            )

        #self.llm1 = LlamaCpp( 
        #model_path="/devdata/home/user/panyongcan/Project/llama.cpp/llama2/13B/ggml-model-q4_0.bin", callbacks=[QueueCallback(self.q)], verbose=True, n_gpu_layers=n_gpu_layers, n_ctx=4096,temperature=0.5,top_k=40,top_p=0.9,max_tokens=2000,max_length=2000,suffix='[INST]',eps=1e-5
        #    )
        # Initialize the LLM we'll be using
        #llm = OpenAI(
        #    streaming=True, 
        #    callbacks=[QueueCallback(q)], 
        #    temperature=0
        #)
        #import ipdb
        #ipdb.set_trace()

        # Create a funciton to call - this will run in a thread
        self.refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template=refine_prompt_template,
        )
        self.initial_qa_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template=initial_qa_template,
        )
        self.chain_type_kwargs = {"question_prompt": self.initial_qa_prompt, "refine_prompt": self.refine_prompt}
    def stream(self,input_text) -> Generator:
        def task():
            #myprompt = simple_prompt.format(question=input_text)
            resp = self.llm(input_text)
            #resp = self.llm(myprompt)
            self.q.put(self.job_done)
        t = Thread(target=task)
        t.start()

        content = ""
        while True:
            try:
                next_token = self.q.get(True, timeout=1)
                if next_token is self.job_done:
                    break
                content += next_token
                yield next_token, content
            except Empty:
                continue

    #def stream1(self,input_text) -> Generator:
    #    def task():
    #        #myprompt = simple_prompt.format(question=input_text)
    #        resp = self.llm1(input_text)
    #        #resp = self.llm(myprompt)
    #        self.q.put(self.job_done)
    #    t = Thread(target=task)
    #    t.start()

    #    content = ""
    #    while True:
    #        try:
    #            next_token = self.q.get(True, timeout=1)
    #            if next_token is self.job_done:
    #                break
    #            content += next_token
    #            yield next_token, content
    #        except Empty:
    #            continue
    #    #pass

    def qa_stream(self,context,input_text) -> Generator:
        #self.loader = TextLoader(file_path)
        #self.documents = self.loader.load()
        #self.texts = self.text_splitter.split_documents(self.documents)
        #import ipdb
        #ipdb.set_trace()
        self.texts = self.text_splitter.split_text(context)
        #self.docsearch = FAISS.from_documents(self.texts, self.embeddings)
        self.docsearch = FAISS.from_texts(self.texts, self.embeddings)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="refine", 
            retriever=self.docsearch.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs=self.chain_type_kwargs)
    
        # Create a Queue
        def task():
            #resp = llm(input_text)
            resp = self.qa(input_text)
            self.q.put(self.job_done)
    
        # Create a thread and start the function
        t = Thread(target=task)
        t.start()
    
        content = ""
    
        # Get each new token from the queue and yield for our generator
        while True:
            try:
                next_token = self.q.get(True, timeout=1)
                if next_token is self.job_done:
                    break
                content += next_token
                yield next_token, content
            except Empty:
                continue

if __name__ == "__main__":
    question = "适合机器人建厂的产业园区有哪些？"
    #myprompt = prompt.format(question=question)
    #for next_token, content in stream(myprompt):
    mymodel = mymodel()
    context = """
早在2017年，常州机器人产业园不仅
有安川、铭赛、金石、快克、邀博、纳博特斯克等核心工业机器人企业，还拥有纳恩博、高尔登、钱医疗等高端服务业机器人企业，一批机器人核心零部件的配
套企业也在此加速聚集。\n\n\n\n安川投资4500万美元的常州第三工厂也在2017年开工建设，这让安川常州工厂年产能力达到1.8万台。\n\n\n\n该产业园在中商
产业研究院整理的“2018年中国机器人产业园区综合发展实力TOP10榜单”中排名第十。\n\n\n\nNO.11\n\n山东西部智能机器人产业园是全国最大的智能服务机器
人生产基地。该产业园主要进行智能机器人平台的研发、设计、生产，同时进行智能科技孵化器建设，为中小科技公司创业提供所需的软硬件设施与服务，以自
身优势和综合平台、服务带动、培育和孵化其他智能科技企业，推进菏泽机器人产业的发展。\n\n\n\n该项目总投资10亿元，规划建筑面积8万平方米。投资企业
拥有50余项专利，其中12项技术处于国内领先水平。项目建成后，年可生产应用服务机器人2000台、自助体检设备2000台，实现产值1亿元。\n\n\n\nNO.12\nNO.
12\n\n深圳南山机器人产业园深圳首个以机器人为主体的产业园。早在2014年，南山区涉及专业机器人的企业就有约35家，从业人数约1.5万人，总产值超过50亿
元（包括其它产品的产值）、出口值达22.5亿元，涌现出固高科技、雷赛智能、众为兴、英威腾、大疆科技等一批研发实力强，掌握机器人及智能设备核心技术
的企业。\n\n\n\n到了2018年，深圳市机器人产业总产值增长至近1200亿元，南山区机器人产业贡献最大，占比近40%。\n\n\n\nNO.13\n\n碧桂园博智林机器人
谷位于佛山顺德区，2019年9月签约，规划占地面积10平方公里，碧桂园计划五年内投入至少800亿元，预计在2023年建成，规划总部基地、智能制造、机器人学
院及科研院校、机器人科研服务支撑平台以及李泽湘-博智林创业园区等五个组团，并引进10000名全球顶级机器人专家及研究人员，打造从机器人人才培养、核
心技术和本体研发，到核心零部件和本体生产制造、各类场景系统集成和实践应用的全产业链服务平台，为机器人产业提供全方位的支持。\n\n湖南工业机器人
产业园位于长沙雨花经济开发区，这是目前全省唯一授牌的工业机器人产业示范园区。\n早在2016年10月，哈南机器人产业园已集聚机器人企业83家，2015年销
售收入突破5亿元，预计2020年机器人产业规模可达到60亿元，正快步向国家级机器人产业示范基地进军。\n\n\n\nNO.19\n\n沈抚新城机器人产业基地规划用地5
平方公里，产业定位是重点发展机器人及智能装备产业。基地2013年就有机器人和智能产业在建项目3个，投资总额42亿元，包括罕王微谷、省电子研究设计院、
申江万国数据库；在谈和重点推进的项目有12个，投资总额120亿元，包括金地绿利机器人产业园、香港纽克瑞森集团自动码垛机器人、自动装箱机器人、成都佳
士科技焊接机器人、厦门思尔特机器人、马丁路德机器人、凯尔达数控切割、香港曼罗兰与美程在线合作的机器人智能印刷项目等。\n\n\n\n2017年末，产业基
地培育产值超亿元机器人企业50家，年实现机器人产业产值500亿元。计划用5年时间，将沈抚新城打造成国内机器人产业技术研发、生产的重点集聚区，2030年
建成国内最大、国际有影响的机器人产业基地。\n\n\n\nNO.20\n\n安徽省合肥机器人产业园位于合肥市高新区，由中国科技大学、合肥高新区、安徽国购集团共
同投资打造，总投资40亿元人民币，总占地370亩，建筑面积2.5万平方公尺，包含机器人启动区、孵化加速区、产业聚集区三部分。
    """
    #for next_token, content in mymodel.qa_stream(context, "适合机器人建厂的产业园区有哪些？请给出具体列表和原因"):
    mystream = mymodel.qa_stream(context, "适合机器人建厂的产业园区有哪些？请给出具体列表和原因")
    for next_token, content in mystream:
        #print(next_token)
        print(content)
    context = """
名单来看，此次上榜的企业数量达243家，其中逾10家半导体企业在列，包括得瑞领新、泽石科技、北京华弘、大唐微电子、北京燕东微电子、清微智能、北京
烁科中科信等。

燕东微电子成立于1987年，是一家集芯片设计、晶圆制造和封装测试于一体的半导体企业

据官方消息显示，燕东微电子在北京、遂宁分别有一条8英寸晶圆生产线和一条6英寸晶圆生产线；在北京拥有一条12英寸晶圆厂在建中

。据官方消息显示，清微智能以三颗芯片量产，数百万颗芯片落地应用，成为全球商业应用规模最大的可重构计算芯片企业。


北京华弘成立于1998年，隶属中国电子信息产业集团，是国内最大的智能卡和集成电路设计、制造企业之一，是专业从事大规模集成电路设计、开发和应用的高
新技术企业，专注数字身份认证、物联网安全、移动互联等领域

其中，海思半导体以303亿元高居榜首，展讯与锐迪科合并后总销售额达125亿，低于此前预期的20亿美元。中兴微电子则以56亿元的销售额排名第三

中芯国际目前是国内规模最大，制造工艺最先进的晶圆制造厂

华力是中国大陆第一条全自动12英寸集成电路Foundry生产线，工艺水平达到55-40-28nm技术等级，月产能3.5万片。华力采用代工模式，为设计公司、IDM公司和
其他系统公司代工逻辑和闪存芯片。

功率半导体器件封测厂商， 中国第一批国家鼓励的集成电路企业。 目前具备年封装及测试各类功率器件12亿只的能力。

中国半导体功率器件五强企业，于2001 年3月在上海证券交易所上市，为国内功率半导体器件领域首家上市公司

中国电子科技集团第五十五研究所下属单位，致力于高性能半导体硅外延片的研发、设计、制造和加工，多年来市场占有率一直稳居第一，为国内领先的硅外延
材料供应商。

系中央企业北京有色金属研究总院全资子公司，主要从事硅和其他电材料的研究、开发、生产与经营，2016年1月，自主研发的“200mm重掺硅单晶抛光片技术”荣
获中国有色金属工业科学技术一等奖。

由七星电子和北方微电子战略重组而成，是目前国内集成电路高端工艺装备的龙头企业。拥有半导体装备、真空装备、新能源锂电装备及精密元器件四个事业群
。

半导体上游原材料中硅片代表企业有中环股份、SK海力士、环球晶圆等，光刻胶企业有晶瑞股份、陶氏化学、科华微电子、旭成化等;封装材料有陶氏杜邦、宏昌
电子等

从代表性企业的所属地分布来看，江苏省是半导体产业代表性企业的集中地，华润微、南大光电、江化微等半导体企业均分布在江苏省。与此同时，浙江省、上
海市、北京市、广东省均有代表性企业分布。


汇总中国半导体行业上市公司2021年相关业务业绩情况，由于身处产业链环节不同、企业自身业务，模式差异，企业盈利能力差别较大，其中从事集成电路生产
销售的臻镭科技相关业务毛利高达80%以上，而从事半导体原材料生产销售的有研新材业务毛利率仅为3.78%

全球半导体公司市值百强出炉，中国公司超半数！前6家公司市值约等于2021年上海+北京+深圳GDP，中芯国际跃居A股首位 _ 证券时报网

当前，半导体已成为高铁、核能、5G、手机、电脑等大多数高科技产业的基石，同时也成为衡量国家科技发展水平的核心指标之一，中国半导体产业迎来高速发
展阶段

。放眼全球，各国（地区）半导体产业规模如何，半导体公司市占率如何、公司规模排名如何？

从全球市占率（营业收入/全球半导体公司营收）来看，排名前十半导体公司市占率接近55%
国内半导体在全球处于什么位置？为保证数据完整性，数据宝以市值来对比（截至8月5日，全部换算为人民币）中国大陆、中国台湾、美国、韩国、日本、德国
、荷兰、挪威等国家（地区）的半导体公司。


按照市值排名，“全球半导体市值百强”的入围门槛由去年末的311.51亿元降低至当前的235.51亿元；位居前六位的公司市值均超过1万亿元，这6家公司的市值合
计11.71万亿元，与2021年上海+北京+深圳的GDP基本持平；英伟达、台积电、阿斯麦、博通稳居前四位，市值均超过1.5万亿元

。中芯国际、北方华创、韦尔股份、紫光国微、三安光电等5股市值均超过千亿，位居全球百强榜的第20至第第35名之间，仅紫光国微因股价跌幅较窄，名次有所
上升

整体来看，国内半导体公司在全球市值、市占率排名相对一般

。国产半导体公司未来将如何发展？中金公司、中信证券均认为，随着美国芯片法案的决议，以及中国大陆对电子制造、终端品牌和市场等需求优势下，我国半
导体产业链有望加速发展。
    """
    for next_token, content in mymodel.qa_stream(context, "北京市的半导体企业有哪些？"):
        #print(next_token)
        print(content)
    for next_token, content in mymodel.stream("北京市的半导体企业有哪些？"):
        #print(next_token)
        print(content)
