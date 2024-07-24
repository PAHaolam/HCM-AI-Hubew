from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import os
import faiss
import json
import torch
from transformers import AutoModel
from translate import translate_vietnamese_to_english  # Import the function
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cung cấp tệp tĩnh từ thư mục "static"
app.mount("/static", StaticFiles(directory="static"), name="static")

IMAGE_FOLDER = "images"

model_jina = AutoModel.from_pretrained(
    'jinaai/jina-clip-v1', trust_remote_code=True)

idx = faiss.read_index("faiss_normal_jina.bin")
id2imgfile = json.load(open('id2imgfile_jina.json'))
for key in id2imgfile:
    # Tách giá trị của value để lấy phần cuối cùng sau dấu "/"
    id2imgfile[key] = '/'.join(id2imgfile[key].split('/')[-2:])


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/display_images")
async def display_images(query: str = Form(...), k: int = Form(...)):
    query = translate_vietnamese_to_english(query)
    text_embedding = model_jina.encode_text(query).reshape((1, -1))
    _, indices = idx.search(text_embedding, k)
    retrieved_images = [
        f'{IMAGE_FOLDER}/{id2imgfile[str(i)]}' for i in indices[0]]

    img_htmls = []
    print(len(retrieved_images))
    for file_name in retrieved_images:
        print(file_name)
        image = Image.open(file_name)
        buffered = io.BytesIO()
        image.save(buffered, format="WEBP")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_html = f'''
        <div>
            <p>{"/".join(file_name.split('/')[-2:])}</p>
            <img src="data:image/webp;base64,{img_str}" alt="Image" />
        </div>
        '''
        img_htmls.append(img_html)

    return JSONResponse(content={"image_data": "".join(img_htmls)})
