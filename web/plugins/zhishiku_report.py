from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,CellStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
import reportlab.lib.fonts
import json
pdfmetrics.registerFont(TTFont('SimHei', './fonts/SimHei.ttf'))

stylesheet=getSampleStyleSheet()
stylesheet.add(ParagraphStyle(fontName='SimHei', name='hei', leading=20, fontSize=12))


reportlab.platypus.tables.CellStyle.fontname='SimHei'
file_path = ''
report = None
def report(report_data):
    global file_path
    report_data = eval(report_data)
    #file_path ="industry_report.pdf"
    # 创建PDF文档的基础类实例
    #import ipdb
    #ipdb.set_trace()
    biaoti = report_data[0]['content']
    file_path = f"report/{biaoti}.pdf"
    #report = SimpleDocTemplate(file_path, pagesize=letter)
    #keys = ['标题','摘要','数据','结论']
    res = []
    import ipdb
    ipdb.set_trace()
    for da in report_data:
        #raw_data = report_data[key]
        da_content,da_type = da['content'],da['type']
        sub_res = create_report_content(da_content,da_type)
        res.extend(sub_res)
    return res
def build(res):
    report = SimpleDocTemplate(file_path, pagesize=letter)
    report.build(res)
    return file_path
# 创建PDF的内容
def create_report_content(da_content,da_type):
    # 创建一个容器，用于添加报告的各个部分
    elements = []

    if da_type == '标题':
        # 报告标题
        #title = "2023年产beijing 业报告"
        #elements.append(Paragraph(title, style=stylesheet['Heading1']))
        elements.append(Paragraph(da_content, style=stylesheet['hei']))
        elements.append(Spacer(1, 12))  # 添加一个空行

    elif da_type == '子标题':
        ## 报告子标题
        subtitle = "深度分析与未来展望"
        elements.append(Paragraph(subtitle, style=stylesheet['hei']))
        elements.append(Spacer(1, 6))  # 添加一个空行

    elif da_type == '摘要':
        # 报告摘要
        #abstract = "本报告对产业发展趋势进行了深度分析，并提供了对未来市场的展望。"
        #elements.append(Paragraph(abstract, style=stylesheet['Normal']))
        elements.append(Paragraph(da_content, style=stylesheet['hei']))
        elements.append(Spacer(1, 12))  # 添加一个空行

    elif da_type == '数据':
        if 'data_type' in da_content:
            return []
        data_type = da_content['data_type']
        data_content = da_content['content']
        if not data_content or data_content==[[]]:
            return []
        import ipdb
        ipdb.set_trace()
        if data_type == '分布':
            table = Table(data_content)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'SimHei'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))  # 添加一个空行
        elif data_type == '数量':
            elements.append(Paragraph(data_content, style=stylesheet['hei']))

    elif da_type == '结论':
        # 报告结论
        #conclusion = "总体来看，互联网、人工智能和生物科技是未来最有增长潜力的产业。"
        elements.append(Paragraph(da_content, style=stylesheet['hei']))
    elif da_type == '正文':
        # 报告正文
        elements.append(Paragraph(da_content, style=stylesheet['hei']))

    return elements

if __name__ == "__main__":
    f = open('json_file/report.json','r')
    datas = json.load(f)
    for data in datas:
        da = data['output']
        #biaoti = da[0]['content']
        # 创建PDF文档的基础类实例
        #report = SimpleDocTemplate(f"report/{biaoti}.pdf", pagesize=letter)
        # 调用create_report_content函数，并将结果添加到报告中
        #report_elements = create_report_content(da)
        #report.build(report_elements)
        file_path = report(da)
        print(file_path)

