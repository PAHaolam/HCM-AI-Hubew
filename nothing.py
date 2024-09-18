import json
import os

for video_json in os.listdir('media-info'):
    # Đọc nội dung file và loại bỏ ký tự lạ
    with open(f'media-info/{video_json}', 'r', encoding='utf-8') as f:
        data = json.loads(f.read().replace('►', ''))

    # Kiểm tra dữ liệu
    print(data['watch_url'])
