# Giả sử bạn đã load mô hình
import open_clip

# Load model từ open_clip
model_name = "nllb-clip-large-siglip"  # hoặc model bạn đang dùng
pretrained = "mrl"  # Tên của mô hình pretrained (nếu có)

print("Loading model ...")
# Load model và preprocess
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)