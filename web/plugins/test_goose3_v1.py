from goose3 import Goose
from goose3.text import StopWordsChinese
g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
html = "https://news.sina.com.cn/c/xl/2023-11-22/doc-imzvmaui9349189.shtml"
html = "https://www.beijing.gov.cn/fuwu/ysfw/jjjhh/cybj/"
article = g.extract(url=html)
print(article.cleaned_text)
