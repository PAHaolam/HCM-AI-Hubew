from fastapi import FastAPI, Request, Form
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
import clip
import open_clip

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cung cấp tệp tĩnh từ thư mục "static"
app.mount("/static", StaticFiles(directory="static"), name="static")

IMAGE_FOLDER = "images"

# Tải các mô hình sẵn có
model_jina = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
model_xlm, _, preprocess_xlm = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained='frozen_laion5b_s13b_b90k')
model_siglip, _, preprocess_siglip = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')

# Khởi tạo faiss và id2imgfile tương ứng cho từng mô hình
faiss_indices = {
    'jinaai/jina-clip-v1': faiss.read_index("faiss_normal_jina.bin"),
    'xlm-roberta-large-ViT-H-14': faiss.read_index("faiss_xml_ViT.bin"),
    'ViT-SO400M-14-SigLIP-384': faiss.read_index("faiss_ViT_SigLIP.bin")
}

id2imgfiles = {
    'jinaai/jina-clip-v1': json.load(open('id2imgfile_jina.json')),
    'xlm-roberta-large-ViT-H-14': json.load(open('id2imgfaiss_xml_ViT.json')),
    'ViT-SO400M-14-SigLIP-384': json.load(open('id2imgfaiss_ViT_SigLIP.json'))
}

# Điều chỉnh đường dẫn trong id2imgfiles để chỉ lấy hai thành phần cuối
for model_name in id2imgfiles:
    for key in id2imgfiles[model_name]:
        id2imgfiles[model_name][key] = os.path.join(*id2imgfiles[model_name][key].split('/')[-2:])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": list(faiss_indices.keys())})

@app.post("/display_images")
async def display_images(query: str = Form(...), k: int = Form(...), model_names: list[str] = Form(...)):
    query = translate_vietnamese_to_english(query)

    image_count = {}

    for model_name in model_names:
        idx = faiss_indices[model_name]
        id2imgfile = id2imgfiles[model_name]
        if model_name == 'jinaai/jina-clip-v1':
            text_embedding = model_jina.encode_text(query)
            text_embedding = text_embedding
        else:
            model = model_xlm if model_name == 'xlm-roberta-large-ViT-H-14' else model_siglip
            tokenizer = open_clip.get_tokenizer(model_name)
            text_tokens = tokenizer([query])
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokens)

        text_embedding = text_embedding.reshape((1, -1))
        _, indices = idx.search(text_embedding, k)
        
        # Update image count
        for idx in indices[0]:
            img_path = id2imgfile[str(idx)]
            if img_path in image_count:
                image_count[img_path] += 1
            else:
                image_count[img_path] = 1

    # Categorize images by tier
    tiered_images = {f'Tier {i+1}': [] for i in range(len(faiss_indices))}
    for img_path, count in image_count.items():
        tier = len(faiss_indices) - count
        tiered_images[f'Tier {tier+1}'].append(img_path)

    img_htmls = []
    for tier, images in tiered_images.items():
        if images:
            img_htmls.append(f"<div class='tier'><h2>{tier}</h2>")
            for file_name in images:
                full_path = os.path.join("C:\\Users\\ADMIN\\Documents\\HCM-AI-2024\\HCM-AI-Hubew", IMAGE_FOLDER, file_name.replace('/', os.sep))
                image = Image.open(full_path)
                buffered = io.BytesIO()
                image.save(buffered, format="WEBP")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                img_html = f'''
                <div class="image">
                    <p>{file_name}</p>
                    <img src="data:image/webp;base64,{img_str}" alt="Image" />
                </div>
                '''
                img_htmls.append(img_html)
            img_htmls.append("</div>")  # Kết thúc thẻ <div> của tier

    return JSONResponse(content={"image_data": "".join(img_htmls)})
