import json

id2imgfile = json.load(open('id2imgfile_jina.json'))
for key in id2imgfile:
    # Tách giá trị của value để lấy phần cuối cùng sau dấu "/"
    id2imgfile[key] = '/'.join(id2imgfile[key].split('/')[-2:])

a =id2imgfile['0']
print(a)
