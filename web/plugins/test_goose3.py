import asyncio
from goose3 import Goose
from goose3.text import StopWordsChinese
import aiohttp

g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def parse_article(url):
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, url)
        #print(html)
        #g = Goose()
        #g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
        article = g.extract(raw_html=html)
        return article.cleaned_text

async def main(urls):
    tasks = [parse_article(url) for url in urls]
    articles = await asyncio.gather(*tasks)
    #for article in articles:
    #    print(article)
    return articles

urls = [
    "https://news.sina.com.cn/c/xl/2023-11-22/doc-imzvmaui9349189.shtml",
    "https://www.tsinghua.edu.cn/info/1175/108010.htm",
    # 更多URLs
]

articles = asyncio.run(main(urls))
import ipdb
ipdb.set_trace()

