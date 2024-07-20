import json
import os

import chromadb
import vertexai
from exa_py import Exa
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vertexai.generative_models import ChatSession, GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from json_from_r import json_from_r

my_sec = os.environ['EXA_API_KEY']
exa = Exa(my_sec)

def get_vertex_embeddings(texts, embedding_model):
  embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
  all_embeddings = []

  # Process texts in batches of 20
  for i in range(0, len(texts), 20):
      batch = texts[i:i+20]
      batch_embeddings = embedding_model.get_embeddings(batch)
      all_embeddings.extend([embedding.values for embedding in batch_embeddings])

  return all_embeddings

def get_answer_from_llm(query, exa, embedding_model, model):
  query = input()
  
  result = exa.search_and_contents(
    query,
    type="neural",
    num_results=10,
    text=True,
    start_published_date="2024-07-01T04:00:01.000Z"
  )
  res =  str(result)
  res_li =res.split('\n')
  
  js = json_from_r(res_li)
  
  for j in js:
    print(j['url'])
  
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      separators=["\n\n", "\n", ".", "!"],
      chunk_overlap=200,
      length_function=len,
      is_separator_regex=False,
  )
  
  texts = []
  for j in js:
    meta = [{"title":j['title'], "url":j["url"]}]
    splits = text_splitter.create_documents([json.dumps(j['text'])],metadatas=meta)
    texts+=splits
  
  key_loc = "dynamic-sun-429817-b4-9f228c3cdf1c.json"
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_loc
  
  vertexai.init(project='dynamic-sun-429817-b4', location="us-central1")
  
  
  
  texts_ind = [text.page_content for text in texts]
  embeddings = get_vertex_embeddings(texts_ind)
  print(len(embeddings))
  
  client = chromadb.Client()
  if client.count_collections()>0:
    client.delete_collection("docs")
  collection = client.create_collection(name = "docs")
  
  count = collection.count()
  
  docs_splits = []
  metadata = []
  ids = [str(i) for i in range(count, count + len(texts))]
  for text in texts:
    docs_splits.append(text.page_content)
    metadata.append(text.metadata)
  
  collection.add(
      ids=ids,
      embeddings=embeddings, # type: ignore
      documents=docs_splits,
      metadatas=metadata,
  )
  
  query_embedding = get_vertex_embeddings([query])
  
  ret = collection.query(query_embeddings=[query_embedding[0]],n_results=5)
  
  context = 'context:'
  for i in range(5):
    t = {}
    t['title'] = ret['metadatas'][0][i]['title'] # type: ignore
    t['url'] = ret['metadatas'][0][i]['url'] # type: ignore
    t['text'] = ret['documents'][0][i] # type: ignore
    context += json.dumps(t)
  
  model_id = 'gemini-1.5-flash-001'
  model_id_adv = 'gemini-1.5-pro-001'
  system_instruction=["You are a personal assistant.","You are given related information from the internet as context use if needed", "cite the source url when using the context"] # type: ignore
  model = GenerativeModel(model_id_adv, system_instruction=system_instruction)  # type: ignore
  
  print(context)
  
  
  chat = model.start_chat()
  # res2 = chat.send_message("user: " + query)
  # print("\n" + res2.text )
  
  query_c = context + " user:" + query
  res = chat.send_message(query_c )
  print("\n" + res.text) #type:ignore


