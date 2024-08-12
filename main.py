import json
import os
from typing import Union

import chromadb
import vertexai
from exa_py import Exa
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vertexai.generative_models import ChatSession, GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from json_from_r import json_from_r
from lm import get_answer_from_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

my_sec = os.environ['EXA_API_KEY']
exa = Exa(my_sec)

key_loc = "dynamic-sun-429817-b4-9f228c3cdf1c.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_loc

vertexai.init(project='dynamic-sun-429817-b4', location="us-central1")

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

client = chromadb.Client()

model_id = 'gemini-1.5-flash-001'
model_id_adv = 'gemini-1.5-pro-001'
system_instruction=["You are a personal assistant.","You are given related information from the internet as context use if needed", "cite the source url when using the context"] # type: ignore
model = GenerativeModel(model_id_adv, system_instruction=system_instruction)  # type: ignore
query = "Did the startup exa ai get funded recently"


@app.get("/exa")
def exa_test():
    # result = exa.search(
    #   "are transformers autoregressive models",
    #   type="neural",
    #   num_results=3
    # )
    # res =  str(result)
    # res_li =res.split('\n')
    with open('data.json') as f:
        js = json.load(f)
    return js


@app.post("/search")
async def search(request: Request):
    body = await request.json()
    print("request body:", body['query'] )
    query = body['query']
    result = exa.search_and_contents(
      query,
      type="neural",
      num_results=5,
        summary=True
    )
    res =  str(result)
    res_li =res.split('\n')
    js = json_from_r(res_li)
    return {"message": "Item received", "body": body, "json": js}

    # with open('data.json') as f:
    #     js = json.load(f)
    # return {"message": "Item received", "body": body, "json": js}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# @app.post("/ask")
# async def answer(request: Request):
#     body = await request.json()
#     print("request body:", body['query'] )
#     query = body['query']
#     res = get_answer_from_llm(query, exa, embedding_model, client, model)
#     print(res)
#     return {"message":"Query Received","json":[{"text":res}]}


@app.post("/ask")
async def answer(request: Request):
    body = await request.json()
    query = body['query']

    return StreamingResponse(
        get_answer_from_llm(query, exa, embedding_model, client, model),
        media_type="text/event-stream"
    )

