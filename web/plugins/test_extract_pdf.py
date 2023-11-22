import requests
from bs4 import BeautifulSoup
import pdfplumber
import docx

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def extract_pdf_content(pdf_filename):
    with pdfplumber.open(pdf_filename) as pdf:
        return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_docx_content(docx_filename):
    doc = docx.Document(docx_filename)
    return '\n'.join(p.text for p in doc.paragraphs)

# 爬虫部分
url = 'https://investhere.ipim.gov.mo/wp-content/uploads/documents/invest_guangzhou_1plus1plusn_cn.pdf'  # 替换为您想爬取的URL
import ipdb
ipdb.set_trace()
#response = requests.get(url)
#soup = BeautifulSoup(response.text, 'html.parser')

filename = download_file(url)
data = extract_pdf_content(filename)
#for link in soup.find_all('a', href=True):
#    file_link = link['href']
#    if file_link.endswith('.pdf') or file_link.endswith('.docx'):
#        filename = download_file(file_link)
#        if filename.endswith('.pdf'):
#            print(f'PDF内容: {extract_pdf_content(filename)}')
#        elif filename.endswith('.docx'):
#            print(f'DOCX内容: {extract_docx_content(filename)}')

