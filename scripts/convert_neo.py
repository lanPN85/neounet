import os
import shutil

SOURCE = 'data/NeoDataset-1300'
TARGET = 'data/NeoDataset-1300_pub'

TRAIN_TARGET = os.path.join(TARGET, 'Train')
TEST_TARGET = os.path.join(TARGET, 'Test')

os.makedirs(os.path.join(TARGET, "Test", "images"))
os.makedirs(os.path.join(TARGET, "Test", "label_images"))
os.makedirs(os.path.join(TARGET, "Test", "mask_images"))

os.makedirs(os.path.join(TARGET, "Train", "images"))
os.makedirs(os.path.join(TARGET, "Train", "label_images"))
os.makedirs(os.path.join(TARGET, "Train", "mask_images"))

TRAIN_FILE = os.path.join(SOURCE, 'train.txt')
TEST_FILE = os.path.join(SOURCE, 'test.txt')


groups = [
    (TRAIN_FILE, TRAIN_TARGET),
    (TEST_FILE, TEST_TARGET),
]

for list_file, target in groups:
    file_names = []
    with open(list_file, 'rt') as f:
        for line in f:
            file_names.append(line.strip().split('.')[0])

    for fn in file_names:
        print(fn)
        image_path = os.path.join(SOURCE, 'images', f'{fn}.jpeg')
        mask_path = os.path.join(SOURCE, 'mask_images', f'{fn}.png')
        label_path = os.path.join(SOURCE, 'label_images', f'{fn}.png')

        shutil.copy2(image_path, os.path.join(target, 'images', f'{fn}.jpeg'))
        shutil.copy2(mask_path, os.path.join(target, 'mask_images', f'{fn}.png'))
        shutil.copy2(label_path, os.path.join(target, 'label_images', f'{fn}.png'))
