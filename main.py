from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
import base64
import faiss
import json
import torch
from transformers import AutoModel
from translate import translate_vietnamese_to_english


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cung cấp tệp tĩnh từ thư mục "static"
app.mount("/static", StaticFiles(directory="static"), name="static")

IMAGE_FOLDER = r"G:\.shortcut-targets-by-id\1StdpWNNHw_g3qHkaedeDXNHgz9GzrJ1L\AIC2024_Hubew\Keyframes_TransNetV2"


# Tải các mô hình sẵn có
model_jina = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

# Khởi tạo faiss và id2imgfile tương ứng cho từng mô hình
# file bin: https://drive.google.com/file/d/13UEWcvYTtyT_7hdHwX6grSDCycmJg4va/view?usp=drive_link
jina_faiss_indices = faiss.read_index(r"D:\TransNetV2\jina_index\jina_indices.bin")
# file json: https://drive.google.com/file/d/1-mniCTAX1DrXwOCdnfsx1YXYnlMo6RJk/view?usp=drive_link
id2imgfiles = json.load(open(r"D:\TransNetV2\image_paths\image_paths.json"))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def path2html(indices):
    retrieved_images = []
    for i in indices[0]:
        retrieved_image_path = id2imgfiles[f'{i}']
        retrieved_images.append({'path': retrieved_image_path, 
                                 'idx': i})

    img_htmls = []
    for retrived_image in retrieved_images:
        full_path = os.path.join(IMAGE_FOLDER, retrived_image['path'])
        try:
            image = Image.open(full_path)
        except:
            image = Image.new('RGB', (1280, 720), (0, 0, 0))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_html = f'''
        <div onclick="enlargeImage(this)">
            <!-- id: {retrived_image['idx']:06} -->
            <p>{full_path[-17:]}</p>
            <img src="data:image/jpeg;base64,{img_str}" alt="Image"/>
        </div>
        '''
        img_htmls.append(img_html)
    return img_htmls


@app.post("/display_images")
async def display_images(query: str = Form(...), k: int = Form(...)):
    query = translate_vietnamese_to_english(query)

    text_embedding = model_jina.encode_text(query)
    text_embedding = text_embedding.reshape((1, -1))
    _, indices = jina_faiss_indices.search(text_embedding, k)

    img_htmls = path2html(indices)

    return JSONResponse(content={"image_data": "".join(img_htmls)})


@app.post("/display_images2")
async def display_images(image: UploadFile = File(...), k: int = Form(...)):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))  # Open it as an image using PIL

    img_embedding = model_jina.encode_image(image)
    img_embedding = img_embedding.reshape((1, -1))
    _, indices = jina_faiss_indices.search(img_embedding, k)

    img_htmls = path2html(indices)

    return JSONResponse(content={"image_data": "".join(img_htmls)})


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    buffered = io.BytesIO(contents)  # Open it as an image using PIL
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_html = f'''
    <img src="data:image/jpeg;base64,{img_str}" alt="Image"/>
    '''
    return JSONResponse(content={"image_data": img_html})