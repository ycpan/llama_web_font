import re
import os
import hashlib
import json
import glob
import numpy as np
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('/home/user/panyongcan/project/big_model/bge-m3') 
res = []
#def get_simility_text_source(query,context):
def get_md5(content):
    md5hash = hashlib.md5(content.encode('utf-8'))
    md5 = md5hash.hexdigest()
    return md5
def get_simility_text_source(answer,context):
    #embeddings_1 = model.encode(query, batch_size=12, max_length=8192)['dense_vecs']# If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    answer_li = answer.split('\n')
    context_json = eval(context)
    content_li,source_li = [],[]
    idx2url = {}
    for idx,da in enumerate(context_json,0):
        content = da['content']
        url = da['source']
        idx2url[idx]=url
        sub_content_li = re.split('\n|。',content)
        sub_content_li = [e for e in sub_content_li if len(e) > 2]
        for element in sub_content_li:
            content_li.append(element)
            source_li.append(url)

    assert len(source_li) == len(content_li)
    content_li_embedding = model.encode(content_li,return_dense=True, return_sparse=False, return_colbert_vecs=False,max_length=256)['dense_vecs']
    count_idx = {}
    new_answer_li = []
    md5_2_url = {}
    for sub_answer in answer_li:
        if not sub_answer:
            new_answer_li.append(sub_answer)
            continue
        sub_answer_li_embedding = model.encode([sub_answer],return_dense=True, return_sparse=False, return_colbert_vecs=False,max_length=256)['dense_vecs']
        similarity =  sub_answer_li_embedding @ content_li_embedding.T
        for simility_idx,sub_answer_similarity in enumerate(similarity):
            sub_max_similarity_idx = np.argsort(sub_answer_similarity)[::-1][0]
            sub_max_similarity = max(sub_answer_similarity)
            assert sub_max_similarity == sub_answer_similarity[sub_max_similarity_idx]
            simility_answer = sub_answer
            simility_context = content_li[sub_max_similarity_idx]
            if sub_max_similarity > 0.88:
                #answer_store_url_li[simility_idx].append(idx)
                url = source_li[sub_max_similarity_idx]
                url_md5 = get_md5(url)
                if url_md5 not in count_idx:
                    counter = len(count_idx) + 1
                    count_idx[url_md5] = counter
                    md5_2_url[url_md5] = url
                counter = count_idx[url_md5]
                sub_answer = sub_answer + f'[^{counter}]'
                print(f'answer:{simility_answer}\t\tcontext:{simility_context}')
            new_answer_li.append(sub_answer)
    sorted_count_idx = sorted(count_idx.items(),key= lambda k:k[1])
    suffix = []
    for url_md5,idx in sorted_count_idx:
        url = md5_2_url[url_md5]
        suffix.append(f'[^{idx}]:{url}')
    #for idx in range(len(idx2url)):
    #    url = idx2url[idx]
    #    suffix.append(f'[^{idx}]:url')
    suffix_str = '\n'.join(suffix)
    new_answer = '\n'.join(new_answer_li) + '\n' + suffix_str

                
    #embeddings_1 = model.encode(answer)['dense_vecs']# If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    #embeddings_2 = model.encode(context)['dense_vecs']
    #similarity = embeddings_1 @ embeddings_2.T
    #print(similarity)
    return new_answer
def get_simility_text_source_v1(answer,context):
    #embeddings_1 = model.encode(query, batch_size=12, max_length=8192)['dense_vecs']# If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    answer_li = answer.split('\n')
    context_json = eval(context)
    content_li,source_li = [],[]
    idx2url = {}
    for idx,da in enumerate(context_json,0):
        content = da['content']
        url = da['source']
        idx2url[idx]=url
        content_li.append(content)

    answer_li_embedding = model.encode(answer_li)['dense_vecs']
    answer_store_url_li = len(answer_li) * [[]]
    for idx,content in enumerate(content_li):
        sub_content_li = re.split('\n|。',content)
        sub_content_li = [e for e in sub_content_li if len(e) > 2]
        if not sub_content_li:
            continue
        sub_content_li_embedding = model.encode(sub_content_li)['dense_vecs']
        similarity =  answer_li_embedding @ sub_content_li_embedding.T
        for simility_idx,sub_answer_similarity in enumerate(similarity):
            sub_max_similarity_idx = np.argsort(sub_answer_similarity)[::-1][0]
            sub_max_similarity = max(sub_answer_similarity)
            assert sub_max_similarity == sub_answer_similarity[sub_max_similarity_idx]
            simility_answer = answer_li[simility_idx]
            simility_context = sub_content_li[sub_max_similarity_idx]
            if sub_max_similarity > 0.90:
                answer_store_url_li[simility_idx].append(idx)
                print(f'answer:{simility_answer}\t\tcontext:{simility_context}')
    assert len(answer_store_url_li) == len(answer_li)
    new_answer_li = []
    for sub_answer,sub_answer_url_li in zip(answer_li,answer_store_url_li):
        sub_answer_url_li =sorted(list(set(sub_answer_url_li)))
        for sub_answer_url_idx in  sub_answer_url_li:
            sub_answer = sub_answer + f'[^{sub_answer_url_idx}]'
        new_answer_li.append(sub_answer)
    suffix = []
    for idx in range(len(idx2url)):
        url = idx2url[idx]
        suffix.append(f'[^{idx}]:url')
    suffix_str = '\n'.join(suffix)
    new_answer = '\n'.join(new_answer_li) + '\n' + suffix_str

                
    #embeddings_1 = model.encode(answer)['dense_vecs']# If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    #embeddings_2 = model.encode(context)['dense_vecs']
    #similarity = embeddings_1 @ embeddings_2.T
    #print(similarity)
    return new_answer
def write_json_line(json_li,file_name):
    f = open('./{}'.format(file_name),'w',encoding='utf-8')
    json_str= json.dumps(json_li,ensure_ascii=False,indent=2)
    json_str.replace('},','\n')
    f.write(json_str)
def get_json_file(input_dir):
    input_file_patern = os.path.join(input_dir,'*.json')
    return glob.glob(input_file_patern)
def process(data):
    res = []
    for da in data:
        context = da['context']
        answer = da['answer']
        question = da['question']
        new_answer = get_simility_text_source(answer,context)
        sub_json = {
            'input':question,
            'context':context,
            'output':new_answer
            }
        res.append(sub_json)
    return res

if __name__ == "__main__":
    #data = get_json_file('question_and_answer.json')
    datas_path = get_json_file('raw_json_file')
    res = []
    for data_path in datas_path:
        data = json.load(open(data_path,'r'))
        sub_res = process(data)
        res.extend(sub_res)
    write_json_line(res,'json_file/parase.json')
