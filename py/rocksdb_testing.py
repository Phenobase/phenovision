import os
import rocksdb
from pathlib import Path
import sys
import io
import PIL
from PIL import Image

# Load a directory of raw images into a db
rdb_options = rocksdb.Options(create_if_missing=True)

def rocks_store_img_dir(db_file, img_dir):
  
  db = rocksdb.DB("phenobase_medium.db", rdb_options)
  
  IMG_DIR = "/blue/guralnick/share/phenobase_inat_data/images/medium"
  
  # Limit number of files for a test
  count = 0
  dbkeys = []

  with os.scandir(IMG_DIR) as entries:
      for fpath in entries:
          if fpath.is_file():
              file_path = fpath.path
              fname = Path(file_path).name
              dbkey = fname.encode('utf-8')
              img = open(file_path, 'rb').read()
              if len(img) == 0:
                  print(f"Skipping {fname}")
                  continue
              #db.put(dbkey, img)
              try:
                  i = Image.open(file_path).tobytes("xbm", "RGB")
              except PIL.UnidentifiedImageError:
                  print(f"Skipping {fname}")
                  continue
              db.put(dbkey, i)
              dbkeys.append(dbkey)
              count += 1
              
  return dbkeys


# Extract stored images for processing:

# import os
# import rocksdb
# from pathlib import Path
# import sys
# import io
# import PIL.Image

# rdb_options = rocksdb.Options(create_if_missing=True)
# db = rocksdb.DB("phenobase_medium.db", rdb_options)
# 
# IMG_DIR = "images"

# Example of reading a file from db for manipulation
keys = db.iterkeys()
keys.seek_to_first()
for key in keys:
    # print(key)
    img_data = db.get(key)
    key_str = key.decode('utf-8')
    raw_data = io.BytesIO(img_data)
    try:
        image = PIL.Image.open(raw_data)
    except (PIL.UnidentifiedImageError, OSError):
        print(f"Bad image data {key_str}")
        continue
    # image.save(some_fname)
    # process the image
