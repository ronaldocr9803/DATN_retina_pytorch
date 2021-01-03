from tqdm import tqdm
from split_image import process_one_image
import os
from natsort import natsorted
import glob

if __name__ == "__main__":
    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists('labels'):
        os.makedirs('labels')    
    lst_raster = ['W03_202003311249_RI_RSK_RSKA014702_RGB', 'W05_202003281250_RI_RSK_RSKA003603_RGB']
    for raster in lst_raster:
        print("Process {}...".format(raster))
        lst_raster_img = natsorted([i for i in os.listdir(os.path.join(".","data",raster)) if i[-4:]=='.tif'])
        for raster_img in lst_raster_img:
            print("Process {}".format(raster_img))
            img_path = os.path.join(".","data", raster, raster_img)
            shfPath = os.path.join(".","data", raster, "{}.shp".format(raster))
            process_one_image(img_path, shfPath)
        # os.listdir(glob.glob(os.path.join(".",raster))[0])
        # 
        # for i in range(len())
        # print(lst_raster_img)