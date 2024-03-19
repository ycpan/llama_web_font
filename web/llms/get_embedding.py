from sklearn.metrics.pairwise import cosine_similarity
from plugins.common import settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
model_name = settings.librarys.qdrant.model_path
model_kwargs = {'device': settings.librarys.qdrant.device}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
def get_embedding(content_li):
    if isinstance(content_li,str):
        content_li = [content_li]
    embedding = hf_embeddings.embed_documents(content_li)
    return embedding
def compute_simility_score(embedding1,embedding2):
    score = cosine_similarity(embedding1,embedding2)
    return score
