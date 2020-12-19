python3 -m pip install geopandas
python3 -m pip install rasterio
python3 -m pip install slidingwindow
export PYTHONPATH=.
python3 demo/visualize_single_image1.py \
    --image_path="./data_tiff/W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB_0.tif" \
    --shapefile_path="./data_tiff/W05_202003281250_RI_RSK_RSKA003603_RGB/W05_202003281250_RI_RSK_RSKA003603_RGB.shp" \
    --model_path="./chkpoint_no_aug/csv_retinanet_29.pt" \
    --evaluate=True
