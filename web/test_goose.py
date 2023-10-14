from goose3 import Goose
from goose3.text import StopWordsChinese
# 初始化，设置中文分词
#g = Goose({'stopwords_class': StopWordsChinese})
g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
# 文章地址
#url = 'http://zhuanlan.zhihu.com/p/46396868'
#url = 'http://henan.kjzch.com/nanyang/2022-06-17/819566.html'
url = 'https://www.sheqi.gov.cn/zw/zcfg2zfb/zcwj/content_3475'
# 获取文章内容
article = g.extract(url=url)
# 标题
print('标题：', article.title)
# 显示正文
print(article.cleaned_text)
#print(article.cleaned_text)
