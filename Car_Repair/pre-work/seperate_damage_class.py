import os
import json
import shutil

## 경로 설정
# 읽어올 파일 : 현재는 front bumper를 읽어올 예정
training_front_bumper_image_path = './bumper/train_front_bp_img/'
training_front_bumper_json_path = './bumper/train_front_bp_json/'

training_front_bumper_img_list = os.listdir(training_front_bumper_image_path)
training_front_bumper_json_list = os.listdir(training_front_bumper_json_path)

img_path = training_front_bumper_image_path
json_path = training_front_bumper_json_path
img_list = training_front_bumper_img_list
json_list = training_front_bumper_json_list

# 폴더 생성
# 이미지 파일과 json 파일이 들어갈 폴더 각각 생성
os.makedirs('./damage_species/Breakage/Breakage_img/',exist_ok=True)
os.makedirs('./damage_species/Breakage/Breakage_json/',exist_ok=True)

os.makedirs('./damage_species/Crushed/Crushed_img/',exist_ok=True)
os.makedirs('./damage_species/Crushed/Crushed_json/',exist_ok=True)

os.makedirs('./damage_species/Scratched/Scratched_img/',exist_ok=True)
os.makedirs('./damage_species/Scratched/Scratched_json/',exist_ok=True)

os.makedirs('./damage_species/Separated/Separated_img/',exist_ok=True)
os.makedirs('./damage_species/Separated/Separated_json/',exist_ok=True)


for i in range(len(json_list)):
    with open (json_path + json_list[i], "r") as f:
        json_files = json.load(f)

    # 손상 종류 : damage_species
    damage_species = ['Breakage', 'Crushed', 'Scratched', 'Separated']
    # 개별 json file에 손상 종류를 넣을 list : tmp_damage_species
    tmp_damage_species = []

    # 개별 json파일을 전부 읽어서 각 json파일에 손상 종류를 list로 넣고 중복 삭제
    for j in json_files['annotations']:
        tmp_damage_species.append(j['damage'])
        new_tmp_damage_species = list(set(tmp_damage_species))

    # 손상 종류가 존재하는 리스트 폴더에 손상 값이 존재할 경우 이미지 파일과 json파일 복사
    for k in damage_species:
        if k in new_tmp_damage_species:
            shutil.copyfile(json_path+json_list[i], './damage_species/' + k + '/' + k + '_img/'+ json_files['images']['file_name'])
            shutil.copyfile(img_path+img_list[i], './damage_species/' + k + '/' + k + '_json/'+ json_list[i])