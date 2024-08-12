import json
import os

import chromadb
import vertexai
from exa_py import Exa
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from vertexai.generative_models import ChatSession, GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from openai import AsyncOpenAI
from groq import AsyncGroq



from json_from_r import json_from_r

my_sec = os.environ['EXA_API_KEY']
exa = Exa(my_sec)
groqKey = os.environ['GROQ_API_KEY']

def get_vertex_embeddings(texts, embedding_model):
  # embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
  all_embeddings = []

  # Process texts in batches of 20
  for i in range(0, len(texts), 20):
      batch = texts[i:i+20]
      batch_embeddings = embedding_model.get_embeddings(batch)
      all_embeddings.extend([embedding.values for embedding in batch_embeddings])

  return all_embeddings

def get_context(query, exa, embedding_model, client):
  # query = input()
  
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
  
  texts_ind = [text.page_content for text in texts]
  embeddings = get_vertex_embeddings(texts_ind, embedding_model)
  print(len(embeddings))
  
  # client = chromadb.Client()
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
  
  query_embedding = get_vertex_embeddings([query],embedding_model)
  
  ret = collection.query(query_embeddings=[query_embedding[0]],n_results=5)
  
  context = 'context:'
  for i in range(5):
    t = {}
    t['title'] = ret['metadatas'][0][i]['title'] # type: ignore
    t['url'] = ret['metadatas'][0][i]['url'] # type: ignore
    t['text'] = ret['documents'][0][i] # type: ignore
    context += json.dumps(t)
  
  # model_id = 'gemini-1.5-flash-001'
  # model_id_adv = 'gemini-1.5-pro-001'
  # system_instruction=["You are a personal assistant.","You are given related information from the internet as context use if needed", "cite the source url when using the context"] # type: ignore
  # model = GenerativeModel(model_id_adv, system_instruction=system_instruction)  # type: ignore
  
  return context

async def get_answer_from_llm(query, exa, embedding_model, client, model):

  context = get_context(query, exa, embedding_model, client)
  
  groq_client  = AsyncGroq(api_key=groqKey)
  query_c = context + " user:" + query

  stream = await groq_client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": "You are a personal assistant, answer the questions in 5 to 10 sentences. You are given related information from the internet as context. Cite the sources when using the context with markdown formating of url and title"},
        {"role": "user", "content": query_c}
    ],
    stream=True,
    stop = None
  )
  async for chunk in stream:
    if chunk.choices[0].delta.content is not None:
      yield chunk.choices[0].delta.content


# my_sec = os.environ['EXA_API_KEY']
# exa = Exa(my_sec)


# key_loc = "dynamic-sun-429817-b4-9f228c3cdf1c.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_loc

# vertexai.init(project='dynamic-sun-429817-b4', location="us-central1")


# embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# client = chromadb.Client()

# model_id = 'gemini-1.5-flash-001'
# model_id_adv = 'gemini-1.5-pro-001'
# system_instruction=["You are a personal assistant.","You are given related information from the internet as context use if needed", "cite the source url when using the context"] # type: ignore
# model = GenerativeModel(model_id_adv, system_instruction=system_instruction)  # type: ignore
# query = "Did the startup exa ai get funded recently"
# res = get_answer_from_llm(query, exa, embedding_model, client, model)
# print(res)