
pip install --upgrade pip

pip install pandas
pip install kaggle

rm -rf data/raw/*
kaggle datasets download -d nih-chest-xrays/data -p data/raw/

tar -xf data/raw/data.zip -C data/raw/
rm -f /data/raw/data.zip
mkdir data/raw/images
mv images_001/images/*.png data/raw/images/
mv images_002/images/*.png data/raw/images/
mv images_003/images/*.png data/raw/images/
mv images_004/images/*.png data/raw/images/
mv images_005/images/*.png data/raw/images/
mv images_006/images/*.png data/raw/images/
mv images_007/images/*.png data/raw/images/
mv images_008/images/*.png data/raw/images/
mv images_009/images/*.png data/raw/images/
mv images_010/images/*.png data/raw/images/
mv images_011/images/*.png data/raw/images/
mv images_012/images/*.png data/raw/images/
rm -rf images_001
rm -rf images_002
rm -rf images_003
rm -rf images_004
rm -rf images_005
rm -rf images_006
rm -rf images_007
rm -rf images_008
rm -rf images_009
rm -rf images_010
rm -rf images_011
rm -rf images_012

