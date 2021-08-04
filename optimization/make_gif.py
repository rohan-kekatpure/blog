from PIL import Image
import sys
from pathlib import Path
from copy import deepcopy


img_dir = Path(sys.argv[1])
img_paths = img_dir.glob('*.png')
img_paths = sorted(img_paths, key=lambda x: int(x.stem.split('_')[1]))
images = []
for img_path in img_paths:
    img = Image.open(img_path)
    img.load()
    images.append(img)


images[0].save('img.gif', save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)


