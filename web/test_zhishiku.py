zhishiku = None
from plugins.common import error_helper, error_print, success_print
from plugins.common import settings
import threading
import logging
logger = None
try:
    from loguru import logger
except:
    pass
def load_zsk():
    try:
        global zhishiku
        import plugins.zhishiku as zsk
        zhishiku = zsk
        success_print("知识库加载完成")
    except Exception as e:
        logger and logger.exception(e)
        error_helper(
            "知识库加载失败，请阅读说明", r"https://github.com/l15y/wenda#%E7%9F%A5%E8%AF%86%E5%BA%93")
        raise e


if __name__ == '__main__':
    thread_load_zsk = threading.Thread(target=load_zsk)
    thread_load_zsk.start()
if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    #res = zhishiku.zsk[0]['zsk'].find('南阳产业政策')
    #res = zhishiku.zsk[1]['zsk'].find("南阳产业政策")
    #res = zhishiku.zsk[2]['zsk'].find("南阳产业政策")
    res = zhishiku.zsk[6]['zsk'].find("南阳产业政策")
    #res = zhishiku.zsk[3]['zsk'].read_find_content(res[0])
    print(res)
