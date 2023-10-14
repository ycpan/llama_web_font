from .chanye_zhenduan import get_chanyezhenduan
from .qiye_zhongdian import get_qiyezhongdian
from .qiye_jishu import get_qiyejishu
from .qiye_chanpin import get_qiyechanpin
from .qiye_yinru import get_qiyeyinru
from .chanye_chengshi import get_chanyechengshi
from .chanye_paiming import get_chanyepaiming
from .chanye_jiegou import get_chanyejiegou
from .chanye_youshi import get_chanyeyoushi
from .chanye_ruoshi import get_chanyeruoshi
from .chanye_jianyi import get_chanyejianyi
def chanye(content_type):
    chanye = None
    #if 'zhenduan' in chanye_type:
    #content_type = 'get_chanyezhenduan(烟台)'
    chanye = eval(content_type)
    return chanye

