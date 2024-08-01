from transformers import AutoModel, AutoTokenizer
import clip
import torch
import open_clip


# Hàm tải mô hình từ Hugging Face và CLIP
def download_models():
    # Tải mô hình từ Hugging Face
    model_names = [
        'jinaai/jina-clip-v1', 
    ]
    
    for model_name in model_names:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"{model_name} has been downloaded.")

    # Tải mô hình từ CLIP
    clip_models = [
        "ViT-B/32",
    ]
    
    for clip_model in clip_models:
        model, preprocess = clip.load(clip_model)
        print(f"{clip_model} has been downloaded.")
        
    # Tải mô hình từ open_clip
    open_clip_models = [
        ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'), 
        ('ViT-SO400M-14-SigLIP-384', 'webli')
    ]
    
    for model_name, pretrained in open_clip_models:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        print(f"{model_name} ({pretrained}) has been downloaded.")

# Thực hiện tải mô hình
if __name__ == "__main__":
    download_models()
