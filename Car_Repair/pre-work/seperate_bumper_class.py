import os
import json
import shutil
training_bumper_image_path = './bumperimg/'
training_bumper_json_path = './bumperjson/'

training_bumper_img_list = os.listdir(training_bumper_image_path)
training_bumper_json_list = os.listdir(training_bumper_json_path)

os.makedirs('./bumper/train_rear_bp_img/',exist_ok=True)
os.makedirs('./bumper/train_rear_bp_json/',exist_ok=True)
os.makedirs('./bumper/train_front_bp_img/',exist_ok=True)
os.makedirs('./bumper/train_front_bp_json/',exist_ok=True)

for i in range(len(training_bumper_json_list)):
    with open (training_bumper_json_path + training_bumper_json_list[i], "r") as f:
        training_bumper_json = json.load(f)
    try:
        if 'Rear bumper' in training_bumper_json['annotations'][0]['repair'][0]:
            try:                
                shutil.copyfile(training_bumper_image_path+training_bumper_json['images']['file_name'], './bumper/train_rear_bp_img/'+training_bumper_json['images']['file_name'])
                shutil.copyfile(training_bumper_json_path+training_bumper_json_list[i], './bumper/train_rear_bp_json/'+training_bumper_json_list[i])                
            except:
                pass
        elif 'Front bumper' in training_bumper_json['annotations'][0]['repair'][0]:
            try:                
                shutil.copyfile(training_bumper_image_path+training_bumper_json['images']['file_name'], './bumper/train_front_bp_img/'+training_bumper_json['images']['file_name'])
                shutil.copyfile(training_bumper_json_path+training_bumper_json_list[i], './bumper/train_front_bp_json/'+training_bumper_json_list[i])                
            except:
                pass
        pass
    except:
        pass
            