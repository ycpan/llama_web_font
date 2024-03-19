
#from plugins.common import settings
import requests
import json
session = requests.Session()
#cunnrent_setting=settings.librarys.remote
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
proxies = {"http": None,"https": None,}


def find(search_query,step = 0):
    #host = cunnrent_setting.host
    host = "10.0.0.20"
    #url = "http://10.0.0.20:19329/v1/query?request=湖南专精特新政策"
    #url = f"http://{host}:19329/v1/query?request={search_query}"
    url = f"http://{host}:19328/v1/query?request={search_query}"

    payload = ""
    headers = {
          'accept': 'application/json'
    }
    res = session.post(url, headers=headers, proxies=proxies ,json={
					"prompt": search_query,
					"step": step
				},data=payload)
    r = res.text
    res = json.loads(r)
    return res['content']
if __name__ == "__main__":
    query = find('湖南专精特新政策')
    print(query)
