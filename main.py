from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

nothing = 0
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2Templates
templates = Jinja2Templates(directory="templates")

IMAGE_FOLDER = r"D:\TransNetV2\keyframes"


# Tải các mô hình sẵn có
model_jina = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

# Khởi tạo faiss và id2imgfile tương ứng cho từng mô hình
# file bin: https://drive.google.com/file/d/13UEWcvYTtyT_7hdHwX6grSDCycmJg4va/view?usp=drive_link
jina_faiss_indices = faiss.read_index(r"D:\TransNetV2\jina_index\jina_indices.bin")
# file json: https://drive.google.com/file/d/1-mniCTAX1DrXwOCdnfsx1YXYXYnlMo6RJk/view?usp=drive_link
id2imgfiles = json.load(open(r"D:\TransNetV2\image_paths\image_paths.json"))

actual_indices = json.load(open(r"D:\TransNetV2\result_dict.json"))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video-detail", response_class=HTMLResponse)
async def video_detail(request: Request):
    return templates.TemplateResponse("video-detail.html", {"request": request})


def path2html(distances, indices, clickable = True):
    retrieved_images = []
    for i, d in zip(indices[0], distances[0]):
        retrieved_image_path = id2imgfiles[f'{i}']
        retrieved_images.append({'path': retrieved_image_path, 
                                 'idx': i,
                                 'distance': round(d, 3) if isinstance(d, float) else d})

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
        address_btn = f'''<a href="/video-detail/{full_path[-17:-9]}/{retrived_image['idx']}">View more</a>'''
        actual_index = actual_indices[full_path[-17:-9]][full_path[-8:-4]]
        img_html = f'''
        <div class="col-xl-3 col-lg-4 col-md-6 col-sm-6 col-12 mb-5">
            <figure class="effect-ming tm-video-item">
                <img src="data:image/jpeg;base64,{img_str}" alt="Image" class="img-fluid">
                <figcaption class="d-flex align-items-center justify-content-center">
                    <h2>{actual_index if actual_index >= 0 else 'NaN'}</h2>
                    {address_btn if clickable else ''}
                </figcaption>                    
            </figure>
            <div class="d-flex justify-content-between tm-text-gray">
                <span class="tm-text-gray-light">Distance={retrived_image['distance']}</span>
                <span>{full_path[-17:-9]}</span>
            </div>
        </div>
        '''
        img_htmls.append(img_html)
    return img_htmls


@app.post("/display_images")
async def display_images(query: str = Form(...)):
    # query = translate_vietnamese_to_english(query)

    text_embedding = model_jina.encode_text(query)
    text_embedding = text_embedding.reshape((1, -1))
    distances, indices = jina_faiss_indices.search(text_embedding, 20)

    img_htmls = path2html(distances, indices)

    return JSONResponse(content={"image_data": "".join(img_htmls)})


@app.post("/display_images2")
async def display_images(image: UploadFile = File(...), k: int = Form(...)):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))  # Open it as an image using PIL

    img_embedding = model_jina.encode_image(image)
    img_embedding = img_embedding.reshape((1, -1))
    distances, indices = jina_faiss_indices.search(img_embedding, k)

    img_htmls = path2html(distances, indices)

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


@app.post("/nearest_images")
async def nearest_images(idx: str = Form(...)):
    idx = int(idx)
    indices = [[]]
    for i in range(-10, 11):
        indices[0].append(idx+i)
    
    img_htmls = path2html(indices)

    return JSONResponse(content={"image_data": "".join(img_htmls)})


@app.get("/video-detail/{id_video}/{idx}", response_class=HTMLResponse)
async def video_detail(request: Request, id_video: str, idx: str):
    full_path = os.path.join(IMAGE_FOLDER, id2imgfiles[f'{idx}'])
    actual_index = actual_indices[id_video][full_path[-8:-4]]

    idx = int(idx)
    indices = [[]]
    distances = [[]]
    for i in range(-4, 5):
        if i == 0:
            continue
        indices[0].append(idx + i)
        distances[0].append(i)
    img_htmls = path2html(distances, indices, False)

    with open(f'media-info/{id_video}.json', 'r', encoding='utf-8') as f:
        data = json.loads(f.read().replace('►', ''))

    data['watch_url'] = f'{data["watch_url"].replace("watch?v=", "embed/")}?start={int(actual_index/25)}'
    print(data['watch_url'])

    return templates.TemplateResponse("video-detail.html", {
        "request": request,
        "image_data": ''.join(img_htmls),
        "video_url": data['watch_url'],
        "id_video": id_video,
        "idx": idx,
        "actual_idx": actual_index
    })


@app.get("/download_csv/{id_video}/{actual_idx}", response_class=StreamingResponse)
# nothing
async def download_csv(id_video: str, actual_idx: str):
    actual_idx = int(actual_idx)
    indices = [[]]
    for i in range(-40, 50, 10):
        if i == 0:
            continue
        indices[0].append(actual_idx + i)

    # Prepare data for CSV
    csv_data = []
    csv_data.append([id_video, actual_idx])
    for i in indices[0]:
        csv_data.append([id_video, i])

    # Create CSV in memory
    def iter_csv():
        for row in csv_data:
            yield ','.join(map(str, row)) + '\n'

    response = StreamingResponse(iter_csv(), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={id_video}_{actual_idx}_keyframes.csv"
    return response

