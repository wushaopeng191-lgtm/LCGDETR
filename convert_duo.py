"""
DUO COCO JSON → YOLO格式，train 8:2划分 train/val，test直接转换
输出: DUO/
用法: python convert_duo.py
"""
import json, os, shutil, random
from PIL import Image
from tqdm import tqdm

random.seed(0)

BASE = '/home/sr/project/LCGDERT/DUO_raw'
OUT = '/home/sr/project/LCGDERT/DUO'

def convert_json(json_path, img_dir):
    """转换一个COCO JSON，返回 [(img_name, [label_lines]), ...]"""
    with open(json_path) as f:
        data = json.load(f)

    img_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_anns[ann['image_id']].append(ann)

    valid = []; bad_img = 0; bad_ann = 0

    for img_info in tqdm(data['images'], desc=f'转换 {os.path.basename(json_path)}'):
        img_name = img_info['file_name']
        img_w, img_h = img_info['width'], img_info['height']

        try:
            im = Image.open(f'{img_dir}/{img_name}')
            im.verify()
            im = Image.open(f'{img_dir}/{img_name}')
            im.load()
        except:
            bad_img += 1
            continue

        lines = []
        for ann in img_anns[img_info['id']]:
            cls_id = ann['category_id'] - 1
            x, y, w, h = ann['bbox']
            x = max(0, x); y = max(0, y)
            w = min(w, img_w - x); h = min(h, img_h - y)
            if w <= 0 or h <= 0:
                bad_ann += 1
                continue
            cx = (x + w/2) / img_w; cy = (y + h/2) / img_h
            nw = w / img_w; nh = h / img_h
            lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        valid.append((img_name, lines))

    if bad_img: print(f'  跳过 {bad_img} 张损坏图')
    if bad_ann: print(f'  跳过 {bad_ann} 个越界标注')
    return valid

def write_split(img_list, img_dir, split_name):
    """写入一个split的图片和标签"""
    os.makedirs(f'{OUT}/{split_name}/images', exist_ok=True)
    os.makedirs(f'{OUT}/{split_name}/labels', exist_ok=True)
    for img_name, lines in tqdm(img_list, desc=f'写入 {split_name}'):
        shutil.copy2(f'{img_dir}/{img_name}', f'{OUT}/{split_name}/images/{img_name}')
        label = os.path.splitext(img_name)[0] + '.txt'
        with open(f'{OUT}/{split_name}/labels/{label}', 'w') as f:
            f.write('\n'.join(lines))
    print(f'  {split_name}: {len(img_list)} 张')

# ====== train → train/val (8:2) ======
print("处理训练集...")
train_data = convert_json(f'{BASE}/annotations/instances_train.json', f'{BASE}/images/train')
random.shuffle(train_data)
n_train = int(len(train_data) * 0.8)
write_split(train_data[:n_train], f'{BASE}/images/train', 'train')
write_split(train_data[n_train:], f'{BASE}/images/train', 'val')

# ====== test (不划分) ======
print("\n处理测试集...")
test_data = convert_json(f'{BASE}/annotations/instances_test.json', f'{BASE}/images/test')
write_split(test_data, f'{BASE}/images/test', 'test')

# ====== data.yaml ======
abs_out = os.path.abspath(OUT)
with open(f'{OUT}/data.yaml', 'w') as f:
    f.write(f"""train: {abs_out}/train/images
val: {abs_out}/val/images
test: {abs_out}/test/images

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
""")

print(f"\n完成! {OUT}/")
print(f"  train: {n_train} 张")
print(f"  val:   {len(train_data)-n_train} 张")
print(f"  test:  {len(test_data)} 张")
