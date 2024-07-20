import json
import os
from typing import Union

from exa_py import Exa
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from json_from_r import json_from_r

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

my_sec = os.environ['EXA_API_KEY']
exa = Exa(my_sec)

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
    result = exa.search(
      query,
      type="neural",
      num_results=5
    )
    res =  str(result)
    res_li =res.split('\n')
    js = json_from_r(res_li)
    return {"message": "Item received", "body": body, "json": js}

    # with open('data.json') as f:
    #     js = json.load(f)
    # return {"message": "Item received", "body": body, "json": js}
