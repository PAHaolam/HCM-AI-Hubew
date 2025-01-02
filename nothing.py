import json

id_video = "L05_V022"
No = 638

actual_indices = json.load(open(r"D:\TransNetV2\result_dict.json"))
actual_index = int(actual_indices[id_video][f'{No:04}'])

print(actual_index)

with open(f'media-info/{id_video}.json', 'r', encoding='utf-8') as f:
    data = json.loads(f.read().replace('►', ''))
print(f"{data['watch_url']}&t={int(actual_index/25)}")