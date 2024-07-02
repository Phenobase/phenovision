import os
import rocksdb
from pathlib import Path
import sys
import io
import PIL
from PIL import Image

# Load a directory of raw images into a db
#rdb_options = rocksdb.Options(create_if_missing=True)

def rocks_store_img_batch(db_file, img_files, rdb_options, test_dup = True):
  
  db = rocksdb.DB(db_file, rdb_options)
  
  #IMG_DIR = "/blue/guralnick/share/phenobase_inat_data/images/medium"
  
  # Limit number of files for a test
  count = 0
  dbkeys = []

  for fpath in img_files:
      if Path(fpath).is_file():
          file_path = fpath
          fname = Path(file_path).name
          dbkey = fname.encode('utf-8')
          img = open(file_path, 'rb').read()
          if len(img) == 0:
              print(f"Skipping {fname}: can't open!")
              continue
          #db.put(dbkey, img)
          try:
              i = Image.open(file_path)
          except PIL.UnidentifiedImageError:
              print(f"Skipping {fname}: can't open!")
              continue
          except OSError:
              print(f"Skipping {fname}: truncated!")
              continue
          try:
            i = i.tobytes("xbm", "RGB")
          except OSError:
              print(f"Skipping {fname}: truncated!")
              continue
          db.put(dbkey, img)
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
# keys = db.iterkeys()
# keys.seek_to_first()
# for key in keys:
#     # print(key)
#     img_data = db.get(key)
#     key_str = key.decode('utf-8')
#     raw_data = io.BytesIO(img_data)
#     try:
#         image = PIL.Image.open(raw_data)
#     except (PIL.UnidentifiedImageError, OSError):
#         print(f"Bad image data {key_str}")
#         continue
#     # image.save(some_fname)
#     # process the image
