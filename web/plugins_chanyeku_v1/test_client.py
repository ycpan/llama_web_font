import http.client
import json
conn = http.client.HTTPConnection("10.0.0.16:6091")
#payload = "{\n    \"industryCode\" : \"H03\",\n    \"parentAreaName\" : \"河南省\",\n    \"areaName\" : \"郑州市\",\n    \"resourceEndowment\" : [\"COU\",\"户籍人口\"],\n\t\"companyLevel\":1\n}"
#payload = {
#            "industryCode":"H03",
#            "parentAreaName":"河南省",
#            "areaName":"郑州市",
#            "resourceEndowment":["COU","户籍人口"],
#            "companyLevel":1
#    }
#
#
#
#

payload = {
    "industryCode":"H03",
    "parentAreaName":"河南省",
    "areaName":"郑州市",
    "resouceEndowment":["COU","户籍人口"],
    "companyLevel":1
}
payload= json.dumps(payload,ensure_ascii=False)
payload = payload.encode()
headers = { 'Hello': "Tom" }
conn.request("POST", "/enterprise_introduction", payload, headers)
res = conn.getresponse()
data = res.read()
print(data)
print(data.decode("utf-8"))
