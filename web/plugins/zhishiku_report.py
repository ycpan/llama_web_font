from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,CellStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
import reportlab.lib.fonts
pdfmetrics.registerFont(TTFont('SimHei', './fonts/SimHei.ttf'))

stylesheet=getSampleStyleSheet()
stylesheet.add(ParagraphStyle(fontName='SimHei', name='hei', leading=20, fontSize=12))


reportlab.platypus.tables.CellStyle.fontname='SimHei'
def report(report_data):
    file_path ="industry_report.pdf"
    # 创建PDF文档的基础类实例
    report = SimpleDocTemplate(file_path, pagesize=letter)
    keys = ['标题','摘要','数据','结论']
    res = []
    for key in keys:
        raw_data = report_data[key]
        sub_res = create_report_content(key,raw_data,res)
        res.extend(sub_res)
    report.build(res)
    return file_path
# 创建PDF的内容
def create_report_content():
    # 创建一个容器，用于添加报告的各个部分
    elements = []

    # 报告标题
    title = "2023年产beijing 业报告"
    #elements.append(Paragraph(title, style=stylesheet['Heading1']))
    elements.append(Paragraph(title, style=stylesheet['hei']))
    elements.append(Spacer(1, 12))  # 添加一个空行

    ## 报告子标题
    subtitle = "深度分析与未来展望"
    elements.append(Paragraph(subtitle, style=stylesheet['hei']))
    elements.append(Spacer(1, 6))  # 添加一个空行

    # 报告摘要
    abstract = "本报告对产业发展趋势进行了深度分析，并提供了对未来市场的展望。"
    #elements.append(Paragraph(abstract, style=stylesheet['Normal']))
    elements.append(Paragraph(abstract, style=stylesheet['hei']))
    elements.append(Spacer(1, 12))  # 添加一个空行

    # 添加一个表格
    data = [
        ['产业分类', '市场规模', '增长速度'],
        ['互联网', '5000亿', '20%'],
        ['人工智能', '1000亿', '30%'],
        ['生物科技', '800亿', '15%']
    ]
    table = Table(data)
    import ipdb
    ipdb.set_trace()
    #cellstyle = CellStyle(name='nopadding')
    #cellstyle.fontname='SimHei'
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

    # 报告结论
    conclusion = "总体来看，互联网、人工智能和生物科技是未来最有增长潜力的产业。"
    elements.append(Paragraph(conclusion, style=stylesheet['hei']))

    return elements

if __name__ == "__main__":
    # 创建PDF文档的基础类实例
    report = SimpleDocTemplate("industry_report.pdf", pagesize=letter)
    # 调用create_report_content函数，并将结果添加到报告中
    report_elements = create_report_content()
    report.build(report_elements)

