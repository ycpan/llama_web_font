from .chanye_zhenduan import get_chanyezhenduan
from .qiye_zhongdian import get_qiyezhongdian
from .chanye_jishu import get_chanyejishu
from .chanye_chanpin import get_chanyechanpin
from .qiye_yinru import get_qiyeyinru
from .chanye_chengshi import get_chanyechengshi
from .chanye_chengshi import get_quanguopaiming
from .chanye_paiming import get_chanyepaiming
from .chanye_jiegou import get_chanyejiegou
from .chanye_youshi import get_chanyeyoushi
from .chanye_ruoshi import get_chanyeruoshi
from .chanye_jianyi import get_chanyejianyi
from .web_tools import get_weather as get_tianqi
from .web_tools import get_baidu_news as get_xinwen
def chanye(content_type):
    chanye = None
    #if 'zhenduan' in chanye_type:
    #content_type = 'get_chanyezhenduan(烟台)'
    chanye = eval(content_type)
    """
    func = eval(func)
    f = map(func,[{'wd':'烟台'}]
    news = list(f)
    return news[0]
    """
    return chanye

