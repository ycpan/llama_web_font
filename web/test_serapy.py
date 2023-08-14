import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
      "q": "南阳产业政策"
})
headers = {
      'X-API-KEY': 'd50d0b2ff04a3bc6ed8101333204d3d0c3281039',
        'Content-Type': 'application/json'
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
