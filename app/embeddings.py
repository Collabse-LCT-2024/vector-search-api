import time
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
# model.cuda()  # uncomment it if you have a GPU

l_tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
l_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
l_model.eval()


def embed_labse(text):
    t0 = time.time()
    encoded_input = l_tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    t1 = time.time()
    print("tokeniser", t1 - t0)
    t0 = time.time()
    with torch.no_grad():
        model_output = l_model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
    t1 = time.time()
    print("inference", t1 - t0)
    return embeddings[0].cpu().numpy()
