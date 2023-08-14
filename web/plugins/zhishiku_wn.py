import requests,json
from gerapy_auto_extractor import extract_list,extract_detail
from goose3 import Goose
from goose3.text import StopWordsChinese
g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})


session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'
}
def find(search_query):
    #url = 'https://cn.bing.com/search?q={}'.format(search_query)
    url = 'https://www.bing.com/search?q={}&mkt=zh-CN&rdr=1&rdrig=1242209326D94FA38E54B55D255F1D0B'.format(search_query)
    res = session.get(url, headers=headers, verify = False).text


    extracted_data = extract_list(res) # 从html源码中解析出搜索引擎的列表
    #for index, item in enumerate(extracted_data):
    #    """
    #    {'title': '南阳：稳产保供强支撑-河南省人民政府门户网站', 'url': 'https://www.henan.gov.cn/2022/07-08/2483362.html'}
    #    """
    #    #print(item)
    content = get_content(extracted_data)
    return content
   
def get_content(res_li):
    """
    {'title': '南阳市出台《关于加快文旅产业高质量发展的实施意见》_市县...', 'url': 'https://www.henan.gov.cn/2022/11-25/2646103.html'}
    """
    res = []
    len_str = 0
    for da in res_li:
        link = da['url']
        article = g.extract(url=link)
        title = article.title
        cleaned_text = article.cleaned_text
        len_str += len(title)
        len_str += len(cleaned_text)
        res.append(title)
        res.append(cleaned_text)
        if len_str > 2000:
            break
    res =  '\n'.join(res)
    return res
