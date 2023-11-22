import json
import asyncio
#import asyn
import requests
import nest_asyncio
nest_asyncio.apply()
api_endpoint = "http://10.0.0.20:19327/v1/embeddings"
def no_sync_get_output(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    #input_messages = { "prompt": input_str}
    input_messages = { "input": input_str}
    headers = {"Content-Type": "application/json",
               #"Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=input_messages)
    #import ipdb
    #ipdb.set_trace()
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        
        print(len(json.loads(response.text)['data']))
        response_text = json.loads(response.text)['data'][0]['embedding']
    else:
        response_text = []
    return response_text
async def get_output(input_str):
    """
    这个接口使用openai的协议，但是不支持stream
    """
    #input_messages = { "prompt": input_str}
    input_messages = { "input": input_str}
    headers = {"Content-Type": "application/json",
               #"Authorization": f"Bearer {access_token}"
               }
    response = requests.post(api_endpoint, headers=headers, json=input_messages)
    #import ipdb
    #ipdb.set_trace()
    if response.status_code == 200:
        #response_text = json.loads(response.text)["choices"][0]["text"]
        
        print(len(json.loads(response.text)['data']))
        response_text = json.loads(response.text)['data'][0]['embedding']
    else:
        response_text = []
    return response_text
    #return response
async def fetch(input_str):
    #async with session.get(url) as response:
        #return await response.text()
    #async with get_output(input_str) as response:
    response = await get_output(input_str)
        #return await response
    return response

async def parse_article(url):
    async with aiohttp.ClientSession() as session:
        try:
            html = await fetch(session, url)
            #print(html)
            #g = Goose()
            g = Goose({'target_language':'zh_cn','browser_user_agent': 'Version/5.1.2 Safari/534.52.7','stopwords_class': StopWordsChinese})
            article = g.extract(raw_html=html)
        #except client_exceptions.ClientConnectorError:
        except Exception as e:
            print(f"Connection failed for URL: {url}")
            # 在这里处理异常，例如记录日志、返回一个默认值等
            return None  # 或者您想返回的任何东西
        return article.cleaned_text

async def mymain(urls):
    #tasks = [parse_article(url) for url in urls]
    tasks = [fetch(url) for url in urls]
    #if loop is None:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(asyncio.gather(*tasks))
    #articles = await asyncio.gather(*tasks)
    #for article in articles:
    #    print(article)
    #return articles
    return result
def get_urls_content(urls):
    #loop = asyncio.get_event_loop()
    articles = []
    #if loop.is_running():
    #    articles = loop.create_task(mymain(urls))
    #else:
    articles = asyncio.run(mymain(urls))
    #articles = asyncio.run(mymain(urls))
    articles = [e for e in articles if e]
    return articles
if __name__ == '__main__':
    input_strs = ["""

    型大城市落户条件。完善城区常住人口500万以上的超大特大城市积分落户政策，精简积分项目，确保社会保险缴纳年限和居住年限分数占主要比例，鼓励取消年度落户名额限制。健全以居住证为载体、与居住年限等条件相挂钩的基本公共服务提供机制，鼓励地方政府提供更多基本公共服务和办事便利，提高居住证持有人城镇义务教育、住房保障等服务的实际享有水平。

""",
    """第二节　健全农业转移人口市民化机制

    完善财政转移支付与农业转移人口市民化挂钩相关政策，提高均衡性转移支付分配中常住人口折算比例，中央财政市民化奖励资金分配主要依据跨省落户人口数量确定。建立财政性建设资金对吸纳落户较多城市的基础设施投资补助机制，加大中央预算内投资支持力度。调整城镇建设用地年度指标分配依据，建立同吸纳农业转移人口落户数量和提供保障性住房规模挂钩机制。根据人口流动实际调整人口流入流出地区教师、医生等编制定额和基本公共服务设施布局。依法保障进城落户农民农村土地承包权、宅基地使用权、集体收益分配权，建立农村产权流转市场体系，健全农户“三权”市场化退出机制和配套政策。

    """,
    """第二十八章　完善城镇化空间布局

    发展壮大城市群和都市圈，分类引导大中小城市发展方向和建设重点，形成疏密有致、分工协作、功能完善的城镇化空间格局。

    第一节　推动城市群一体化发展

    以促进城市群发展为抓手，全面形成“两横三纵”城镇化战略格局。优化提升京津冀、长三角、珠三角、成渝、长江中游等城市群，发展壮大山东半岛、粤闽浙沿海、中原、关中平原、北部湾等城市群，培育发展哈长、辽中南、山西中部、黔中、滇中、呼包鄂榆、兰州－西宁、宁夏沿黄、天山北坡等城市群。建立健全城市群一体化协调发展机制和成本共担、利益共享机制，统筹推进基础设施协调布局、产业分工协作、公共服务共享、生态共建环境共治。优化城市群内部空间结构，构筑生态和安全屏障，形成多中心、多层级、多节点的网络型城市群。

    第二节　建设现代化都市圈

    依托辐射带动能力较强的中心城市，提高1小时通勤圈协同发展水平，培育发展一批同城化程度高的现代化都市圈。以城际铁路和市域（郊）铁路等轨道交通为骨干，打通各类“断头路”、“瓶颈路”，推动市内市外交通有效衔接和轨道交通“四网融合”，提高都市圈基础设施连接性贯通性。鼓励都市圈社保和落户积分互认、教育和医疗资源共享，推动科技创新券通兑通用、产业园区和科研平台合作共建。鼓励有条件的都市圈建立统一的规划委员会，实现规划统一编制、统一实施，探索推进土地、人口等统一管理。

    第三节　优化提升超大特大城市中心城区功能

    统筹兼顾经济、生活、生态、安全等多元需要，转变超大特大城市开发建设方式，加强超大特大城市治理中的风险防控，促进高质量、可持续发展。有序疏解中心城区一般性制造业、区域性物流基地、专业市场等功能和设施，以及过度集中的医疗和高等教育等公共服务资源，合理降低开发强度和人口密度。增强全球资源配置、科技创新策源、高端产业引领功能，率先形成以现代服务业为主体、先进制造业为支撑的产业结构，提升综合能级与国际竞争力。坚持产城融合，完善郊区新城功能，实现多中心、组团式发展。

    第四节　完善大中城市宜居宜业功能

    充分利用综合成本相对较低的优势，主动承接超大特大城市产业转移和功能疏解，夯实实体经济发展基础。立足特色资源和产业基础，确立制造业差异化定位，推动制造业规模化集群化发展，因地制宜建设先进制造业基地、商贸物流中心和区域专业服务中心。优化市政公用设施布局和功能，支持三级医院和高等院校在大中城市布局，增加文化体育资源供给，营造现代时尚的消费场景，提升城市生活品质。

    第五节　推进以县城为重要载体的城镇化建设

    加快县城补短板强弱项，推进公共服务、环境卫生、市政公用、产业配套等设施提级扩能，增强综合承载能力和治理能力。支持东部地区基础较好的县城建设，重点支持中西部和东北城镇化地区县城建设，合理支持农产品主产区、重点生态功能区县城建设。健全县城建设投融资机制，更好发挥财政性资金作用，引导金融资本和社会资本加大投入力度。稳步有序推动符合条件的县和镇区常住人口20万以上的特大镇设市。按照区位条件、资源禀赋和发展基础，因地制宜发展小城镇，促进特色小镇规范健康发展。   如何加强水利建设"""
    ]
    data = []
    for input_str in input_strs*100:
        embedding = no_sync_get_output(input_str)
        data.append(embedding)
    #data = get_urls_content(input_strs* 100)
    print(data)
