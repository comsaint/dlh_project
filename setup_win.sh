pip install -r requirement.txt

kaggle datasets download -d nih-chest-xrays/data -p data/raw/

tar -xf data\raw\data.zip -C data\raw\
del \data\raw\data.zip
mkdir data\processed

# move images and necessary files to 'processed'
move data\raw\images_001\images\*.png data\processed
move data\raw\images_002\images\*.png data\processed
move data\raw\images_003\images\*.png data\processed
move data\raw\images_004\images\*.png data\processed
move data\raw\images_005\images\*.png data\processed
move data\raw\images_006\images\*.png data\processed
move data\raw\images_007\images\*.png data\processed
move data\raw\images_008\images\*.png data\processed
move data\raw\images_009\images\*.png data\processed
move data\raw\images_010\images\*.png data\processed
move data\raw\images_011\images\*.png data\processed
move data\raw\images_012\images\*.png data\processed
move data\raw\*.csv data\processed
move data\raw\*.txt data\processed

# clean up
rmdir data\raw /S /Q
