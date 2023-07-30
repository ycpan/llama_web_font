from langchain.embeddings.huggingface import HuggingFaceEmbeddings
model_name = "/home/user/panyongcan/project/big_model/m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
import ipdb
ipdb.set_trace()
