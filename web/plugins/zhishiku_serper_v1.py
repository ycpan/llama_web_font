import http.client
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import pdfplumber
import docx
import fitz
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import json
import os
import re
import jieba
import numpy as np
from goose3 import Goose
from goose3.text import StopWordsChinese
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from plugins.common import settings, allowCROS
import nest_asyncio
from aiohttp import client_exceptions
nest_asyncio.apply()
model_name = settings.librarys.qdrant.model_path
#SERPER_API_KEY=os.getenv("d50d0b2ff04a3bc6ed8101333204d3d0c3281039")
SERPER_API_KEY="d50d0b2ff04a3bc6ed8101333204d3d0c3281039"
g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
model_kwargs = {'device': settings.librarys.qdrant.device}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
#async def download_file(url):
#    local_filename = url.split('/')[-1]
#    async with requests.get(url, stream=True) as r:
#        with open(local_filename, 'wb') as f:
#            for chunk in r.iter_content(chunk_size=8192):
#                f.write(chunk)
#    return local_filename
def extract_pdf_content(pdf_filename):
    #with pdfplumber.open(pdf_filename) as pdf:
    #    return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
    res = []
    with fitz.open(pdf_filename) as doc:
        for page in doc: # iterate the document pages
            text = page.get_text()
            res.append(text)
    return '\n'.join(res)
def extract_docx_content(docx_filename):
    doc = docx.Document(docx_filename)
    return '\n'.join(p.text for p in doc.paragraphs)
async def download_file(session, url,timeout=3):
    local_filename = url.split('/')[-1]
    async with session.get(url,timeout=timeout) as response:
        if response.status == 200:
            f = await aiofiles.open(local_filename, mode='wb')
            await f.write(await response.read())
            await f.close()
            return local_filename
async def fetch(session, url,timeout=5):
    async with session.get(url,timeout=timeout) as response:
        return await response.text()

async def download_and_extract_file(session, file_link,timeout=3):
    filename = await download_file(session, file_link,timeout)
    article_text = ''
    #import ipdb
    #ipdb.set_trace()
    if filename:
        if filename.endswith('.pdf'):
            print(f'PDF内容:')
            article_text = extract_pdf_content(filename)
            print(article_text[0:100])
        elif filename.endswith('.docx'):
            print(f'DOCX内容:')
            article_text = extract_docx_content(filename)
            print(article_text[0:100])
    return article_text
async def parse_article(url):
    async with aiohttp.ClientSession() as session:
        try:
            article_text = ''
            if url.endswith('.pdf') or url.endswith('.docx'):
                #filename = await download_file(session,url)
                article_text = await download_and_extract_file(session, url,3)
                #import ipdb
                #ipdb.set_trace()
            else:
                html = await fetch(session, url,3)
                #print(html)
                #import ipdb
                #ipdb.set_trace()
                #g = Goose()
                g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
                article = g.extract(raw_html=html)
                article_text = article.cleaned_text
                tasks = []
                if len(article_text) < 200:
                    soup = BeautifulSoup(html, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        file_link = link['href']
                        if file_link.endswith('.pdf') or file_link.endswith('.docx'):
                            file_link = urljoin(url, link['href'])
                            print('pdf or word in html')
                            print(file_link)
                            article_text = await download_and_extract_file(session, file_link,3)
                            if len(article_text) > 200:
                                break
                            #task = asyncio.create_task(download_and_extract_file(session, file_link))
                            #tasks.append(task)
                if tasks:
                    await asyncio.gather(*tasks)

        #except client_exceptions.ClientConnectorError:
        except Exception as e:
            print(f"Connection failed for URL: {url}. Error: {e}")
            # 在这里处理异常，例如记录日志、返回一个默认值等
            return None  # 或者您想返回的任何东西
        #return article.cleaned_text
        if len(article_text) < 50:
            article_text = ''
        return article_text

async def mymain(urls):
    #import ipdb
    #ipdb.set_trace()
    tasks = [parse_article(url) for url in urls]
    #if loop is None:
    #loop = asyncio.get_event_loop()
    #result = loop.run_until_complete(asyncio.gather(*tasks))
    articles = await asyncio.gather(*tasks)
    #for article in articles:
    #    print(article)
    #return result
    return articles
def get_urls_content(urls):
    #loop = asyncio.get_event_loop()
    articles = []
    #if loop.is_running():
    #    articles = loop.create_task(mymain(urls))
    #else:
    #import ipdb
    #ipdb.set_trace()
    #articles = asyncio.run(mymain(urls))
    #urls = ['https://www.ndrc.gov.cn/xxgk/zcfb/ghwb/202103/t20210323_1270124.html']
    #articles = asyncio.run(mymain(urls),debug=True)
    articles = asyncio.run(mymain(urls))
    articles = [e for e in articles if e]
    print('articles len is {}'.format(len(articles)))
    return articles
def find(search_query,step = 0):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": search_query
    })
    headers = {
    'X-API-KEY': SERPER_API_KEY,
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    #import ipdb
    #ipdb.set_trace()
    data = res.read()
    data=json.loads(data)
    clean_data = data['organic']
    content = get_content(clean_data)
    related_res = get_related_content(search_query,content)
    #print(data)
    #l=[{'title': "["+organic["title"]+"]("+organic["link"]+")", 'content':organic["snippet"]} for organic in data['organic']]
    #try:
    #    if data.get("answerBox"):
    #        answer_box = data.get("answerBox", {})
    #        l.insert(0,{'title': "[answer："+answer_box["title"]+"]("+answer_box["link"]+")", 'content':answer_box["snippet"]})
    #except:
    #    pass
    #return l,data
    #return content
    return related_res
def get_content(res_li):
    """
    {'title': '《南阳市“十四五”现代服务业发展规划》政策解读', 'link': 'http://henan
    .kjzch.com/nanyang/2022-06-17/819566.html', 'snippet': '顺应产业融合需求，秉承“
    两业并举”，以服务产业升级、提高流通效率为导向，大力发展现代金融、现代物流、科技
    服务、信息服务、中介服务、节能环保服务、 ...', 'date': 'Jun 17, 2022', 'attribu
    tes': {'南阳市人民政府办公室': '2022-06-17'}, 'position': 1}
    """
    res = []
    len_str = 0
    urls = []
    for da in res_li:
        link = da['link']
        urls.append(link)
        #try:
        #    article = g.extract(url=link)
        #    title = article.title
        #    cleaned_text = article.cleaned_text
        #    len_str += len(title)
        #    len_str += len(cleaned_text)
        #    res.append(title)
        #    res.append(cleaned_text)
        #except Exception as e:
        #    print(e)
        ##if len_str > 1500:
        #if len_str > 12000:
        #    break
    #import ipdb
    #ipdb.set_trace()
    res = get_urls_content(urls)
    res =  '\n'.join(res)
    return res

def get_related_content_old(query,content):
    def clean_text(content):
        res = []
        for txt in content.split('\n'):
            if not txt:
                continue
            if len(txt) < 5:
                continue
            res.append(txt.strip())
        return '\n'.join(res)

    #content_li = content.split('。')
    #import ipdb
    #ipdb.set_trace()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
    #texts = text_splitter.split_documents(content)
    content = clean_text(content)
    content_li = text_splitter.split_text(content)
    content_li.append(query)
    embedding = hf_embeddings.embed_documents(content_li)
    score = cosine_similarity([embedding[-1]],embedding[0:-1])
    idxs = score.argsort()
    idxs = idxs[0][::-1]
    res = ['']*len(content_li)
    len_str = 0
    #import ipdb
    #ipdb.set_trace()
    for idx in idxs:
        s = score[0][idx]
        if s < 0.75:
            continue
        sub_content = content_li[idx]
        if not sub_content:
            continue
        if len(sub_content) < 15:
            continue
        len_str += len(sub_content)
        res[idx]=sub_content
        #if len_str > 1000:
        #if len_str > 1500:
        if len_str > 1500:
        #if len_str > 8500:
            break
    final_res = []
    #len_res = len(res)
    for idx,txt in enumerate(res):
        #start = idx - 3 if idx - 3 >= 0 else 0
        #end = idx + 3 if idx + 3 < len_res else len_res - 1
        #useful_count = len([i for i in res[start:end+1] if i])
        #ratio = useful_count / len(res[start:end+1])
        #if ratio > 0.28:
        #    final_res.append(content_li[idx])
        if txt.strip() and len(txt) > 10:
            final_res.append(txt)

    res = '\n\n'.join(final_res)
    #res = '\n'.join(res)
    #return res[0:1700]
    return res[0:2500]
    #return res[0:8700]

def get_related_content(query,content):
    def clean_text(content):
        res = []
        for txt in content.split('\n'):
            if not txt:
                continue
            if len(txt) < 5:
                continue
            res.append(txt.strip())
        return '\n'.join(res)

    #content_li = content.split('。')
    #import ipdb
    #ipdb.set_trace()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
    #texts = text_splitter.split_documents(content)
    content = clean_text(content)
    content_li = text_splitter.split_text(content)
    detail_content_li = []
    detail_content_li_paragraph = []
    for paragraph in content_li:
        for sentence in re.split('\r|\n|,|\.|。|；|;',paragraph):
            if len(sentence) < 2:
                continue
            detail_content_li.append(sentence)
            detail_content_li_paragraph.append(paragraph)
    document_embedding = hf_embeddings.embed_documents(detail_content_li)

    query_li = list(jieba.cut(query))
    query_windows_li = []
    for idx in range(len(query_li)):
        query_windows_li.append(query_li[idx])
        query_windows_li.append(''.join(query_li[idx:idx+2]))
        query_windows_li.append(''.join(query_li[idx:idx+3]))
        query_windows_li.append(''.join(query_li[idx:idx+4]))
    query_li = list(set(query_windows_li))
    print(query_li)
    query_embedding = hf_embeddings.embed_documents(query_li)

    #embedding = hf_embeddings.embed_documents(content_li)
    #score = cosine_similarity([embedding[-1]],embedding[0:-1])
    score = cosine_similarity(query_embedding,document_embedding)
    #idxs_all = score.argsort()
    #res = ['']*len(detail_content_li_paragraph)
    #res_score = [0]*len(detail_content_li_paragraph)

    ##import ipdb
    ##ipdb.set_trace()
    #for index,idxs in enumerate(idxs_all):
    #    #idxs = idxs[0][::-1]
    #    #res = ['']*len(content_li)
    #    len_str = 0
    #    idxs = idxs[::-1]
    #    for idx in idxs[0:3]:
    #        s = score[index][idx]
    #        print('socre:{}\titem:{}\toriginal:{}'.format(s,detail_content_li[idx],query_li[index]))
    #        if s < 0.75:
    #            continue
    #        #sub_content = content_li[idx]
    #        sub_content = detail_content_li_paragraph[idx]
    #        if not sub_content:
    #            continue
    #        if len(sub_content) < 15:
    #            continue
    #        len_str += len(sub_content)
    #        #res[idx]=sub_content
    #        if s > res_score[idx]:
    #            res_score[idx]=s
    #        #if len_str > 1000:
    #        if len_str > 6000:
    #            break
    ##len_res = len(res)
    #res_score_idx = np.argsort(res_score)
    #count = 0
    #res_score_idx = res_score_idx[::-1]
    #for idx in res_score_idx:
    #    txt = detail_content_li_paragraph[idx]
    #    count += len(txt)
    #    if count > 2000:
    #        break
    #    #if txt.strip() and len(txt) > 10:
    #    #    final_res.append(txt)
    #    res[idx]=txt

    #final_res = ''
    #for txt in res:
    #    if len(txt) > 2:
    #        if txt not in final_res:
    #            final_res += txt + '\n\n'
    ##res = '\n\n'.join(final_res)
    ##res = '\n\n'.join(res)
    #res = final_res
    #return res[0:1700]
    score = score.mean(axis=0)
    score_idx = score.argsort()
    score_idx = score_idx[::-1]
    res = []
    res_len = 0
    for idx in score_idx:
        sub_graph = detail_content_li_paragraph[idx]
        if len(sub_graph) < 30:
            continue
        if sub_graph not in res:
            res.append(sub_graph)
            res_len += len(sub_graph)
        if res_len > 2000:
            break
    res_str = '\n'.join(res)
    res_str =  res_str[0:1700]
    return res_str



    #idxs = score.argsort()
    #idxs = idxs[0][::-1]
    #res = ['']*len(content_li)
    #len_str = 0
    ##import ipdb
    ##ipdb.set_trace()
    #for idx in idxs:
    #    s = score[0][idx]
    #    if s < 0.75:
    #        continue
    #    sub_content = content_li[idx]
    #    if not sub_content:
    #        continue
    #    if len(sub_content) < 15:
    #        continue
    #    len_str += len(sub_content)
    #    res[idx]=sub_content
    #    #if len_str > 1000:
    #    #if len_str > 1500:
    #    if len_str > 1500:
    #    #if len_str > 8500:
    #        break
    #final_res = []
    ##len_res = len(res)
    #for idx,txt in enumerate(res):
    #    #start = idx - 3 if idx - 3 >= 0 else 0
    #    #end = idx + 3 if idx + 3 < len_res else len_res - 1
    #    #useful_count = len([i for i in res[start:end+1] if i])
    #    #ratio = useful_count / len(res[start:end+1])
    #    #if ratio > 0.28:
    #    #    final_res.append(content_li[idx])
    #    if txt.strip() and len(txt) > 10:
    #        final_res.append(txt)

    #res = '\n\n'.join(final_res)
    ##res = '\n'.join(res)
    ##return res[0:1700]
    #return res[0:2500]
    ##return res[0:8700]


