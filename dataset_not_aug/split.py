import os
from natsort import natsorted
import shutil
from tqdm import tqdm

if not os.path.exists("./test_RSKA003603_0_"):
    os.makedirs("./test_RSKA003603_0_")

lst_img = natsorted([img for img in os.listdir("./images") if "Img_RSKA003603_0_" in img])
for img in tqdm(lst_img):
    src_img = os.path.join(".","images", img)
    des_img = os.path.join(".", "test_RSKA003603_0_", img)
    shutil.move(src_img, des_img)
# import ipdb; ipdb.set_trace()