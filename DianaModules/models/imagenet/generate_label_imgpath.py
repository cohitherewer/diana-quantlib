

import os 
from pathlib import Path 
import pandas as pd 
imagenet_base_dir =os.path.normpath(str(Path("../../../../../projectdata/datasets/imagenet/small").absolute() ) )
imagenet_train_path =imagenet_base_dir + '/train' 
imagenet_val_path =imagenet_base_dir + '/val' 
classes_file = str(Path("classes.txt").absolute())
# load classes file 
classes_labels = pd.read_csv(classes_file, delimiter=" ", header=None, index_col=0).iloc[: ,:1].to_dict()[1]
assert len(classes_labels.keys()) == 1000 
#for key in classes_labels.keys(): 
#    print(key) 




# generating training value map 
train_val_map = "train_val_map.txt"
validation_val_map = "validation_val_map.txt" 

counter = 0 

with open(train_val_map , 'w') as f: 
    for path, subdirs, files in os.walk(imagenet_train_path):
        for subdir in subdirs: 
            for _,_,files in os.walk(path+f"/{subdir}"): 
                for name in files:

                    f.write(f"{path}/{subdir}/{name} {classes_labels[subdir]}\n")

                    counter +=1 
                
print(f"Written {counter} training lines") 
counter = 0
# generating validation value map 
with open(validation_val_map, 'w') as f: 
    for path, subdirs, files in os.walk(imagenet_val_path):
        for subdir in subdirs: 
            for _,_,files in os.walk(path+f"/{subdir}"): 
                for name in files:

                    f.write(f"{path}/{subdir}/{name} {classes_labels[subdir]}\n")

                    counter +=1 
 
print(f"Written {counter} validation lines") 

# Written 1281167 training lines
# Written 50000 validation lines