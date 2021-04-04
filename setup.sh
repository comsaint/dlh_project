pip install -r requirement.txt

rm -r data/raw/*
kaggle datasets download -d nih-chest-xrays/data -p data/raw/

unzip data/raw/data.zip -d data/raw/
rm /data/raw/data.zip
unzip '/data/raw/*.zip' -d /data/raw/
