from django.shortcuts import render, redirect
from django.contrib import auth
from .models import User, Image
from django.http import HttpResponse, JsonResponse
from pymongo import MongoClient
from config import *
from elasticsearch import Elasticsearch
from django.core.paginator import Paginator

import torch, os, re, cv2, joblib, json, csv
from torch import nn, optim
import segmentation_models_pytorch as smp
import pandas as pd
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import csv
from django.http import HttpResponse, JsonResponse

# 로그 쓰는 함수
# id, main_function, client_ip, searched_word, clicked_repairshop_name, clicked_repairshop_id
def logging(logging_text):
    with open('./django_log.log', 'a') as f:
        f.write(logging_text)
        f.write('\n')

def index(request):
    return render(request, 'index.html')

def login(request):
    if request.method == "GET" :
        return render(request, 'login.html')
    elif request.method == "POST" :
        username = request.POST.get('username')
        password = request.POST.get('password')

        res_data = {}
        if username == '' or password == '':
            res_data['error'] = "모든 칸을 다 입력해주세요."
        else :
            user = User.objects.get(username=username)
            if password == user.password:
                request.session['user'] = user.id
                return redirect('/index')
            else:
                res_data['error'] = "비밀번호가 틀렸습니다."

    return render(request, 'login.html', res_data)

def signUp(request):
    userdb = User.objects.all()
    email_validation = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    phone_number_validation = re.compile('\d{2,3}-\d{3,4}-\d{4}')
    password_validation = re.compile("^(?=.*[A-Za-z])(?=.*\d)(?=.*[$@$!%*#?&])[A-Za-z\d$@$!%*#?&]{8,}$")
    if request.method == 'GET' :
        return render(request, 'signUp.html')
    elif request.method == 'POST' :
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']
        phone_number = request.POST['phone_number']
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']

        res_data = {}
        if username == '' or email == '' or firstname == '' or lastname == '' or phone_number == '' or password1 == '':
            res_data['empty_error'] = '모든 정보가 입력되지 않았습니다.'
        elif password1 != password2:
            res_data['pw_error02'] = "비밀번호가 다릅니다."
        elif userdb.filter(username = username).exists():
            res_data['username_error'] = "이미 존재하는 아이디입니다."
        elif userdb.filter(phone_number = phone_number).exists():
            res_data['phone_number_error02'] = "이미 가입된 연락처입니다."
        elif email_validation.match(email) == None:
            res_data['email_error'] = '@와 .를 포함한 이메일형식이 아닙니다.'
        elif phone_number_validation.match(phone_number) == None:
            res_data['phone_number_error'] = '-를 포함한 연락처 형식이 아닙니다.'
        elif password_validation.match(password1) == None:
            res_data['pw_error01'] = '비밀번호 형식을 확인해주세요.'
        else:
            # user = User_data(
            #     username=username,
            #     password=password1,
            #     email=email,
            #     phone_number=phone_number,
            #     firstname=firstname,
            #     lastname=lastname,
            # )
            # user.save()

            user = User.objects.create_user(
                username=request.POST['username'],
                password=request.POST['password1'],
                email=request.POST['email'],
                first_name=request.POST['firstname'],
                last_name=request.POST['lastname'],
                phone_number=request.POST['phone_number']
            )
            auth.login(request, user)
            res_data['success'] = 'ok'


        return render(request, 'signUp.html', res_data)

def main(request):
    # logging
    # id, main_function, client_ip, searched_word
    logging_text = request.user.username + ',' + 'main' + ',' + request.META.get('REMOTE_ADDR') + ','
    logging(logging_text)
    return render(request, 'main.html')

def side01(request):
    return render(request, 'side01.html')

def side02(request):
    return render(request, 'side02.html')

def show_repairs(request):
    # logging
    logging_text = request.user.username + ',' + 'show_near_repairshop' + ',' + request.META.get('REMOTE_ADDR') + ','
    logging(logging_text)
    # 좌표값 get
    input_lat = float(request.GET.get('input_lat'))
    input_lon = float(request.GET.get('input_lon'))

    # 몽고db 연결
    client = MongoClient(MONGODB_CONFIG['url'])
    db = client['test']

    # 좌표값에서 반경 5km 이내에 있는 데이터 검색 쿼리
    query = {'location':
                 {'$geoWithin':
                      {'$centerSphere': [[input_lon, input_lat], 5 / 6378.1]}}}

    # 휴점, 폐점 데이터를 제외하고 json 형식으로 변환해서 리스트에 삽입
    repair_list = list()
    for doc in db.repair.find(query):
        repair_dict = dict()
        repair_dict['id']= doc['s_id']
        repair_dict['자동차정비업체명'] = doc['s_name']
        repair_dict['자동차정비업체종류'] = doc['s_type']
        repair_dict['주소'] = doc['address']
        repair_dict['좌표'] = doc['location']['coordinates']
        repair_dict['영업상태'] = doc['running']
        repair_dict['운영시작시각'] = doc['open_time']
        repair_dict['운영종료시각'] = doc['close_time']
        repair_dict['전화번호'] = doc['phone_number']
        if repair_dict['영업상태'] == 1:
            repair_list.append(repair_dict)

    return JsonResponse({'list': repair_list})


def marker_click(request):
    s_name = request.GET.get('s_name')
    s_id = request.GET.get('s_id')
    # logging
    # id, main_function, client_ip, searched_word
    logging_text = request.user.username + ',' + 'clicked_repairshop' + ',' + request.META.get('REMOTE_ADDR') + ',' + ',' + s_name + ',' + s_id
    logging(logging_text)
    return HttpResponse(s_name)

def click_log(request):
    user = request.user.username

    # elasticsearch 연결
    end_point = ELASTIC_CONFIG['url']
    es = Elasticsearch(hosts=end_point)

    # 검색 결과 수 제한 수정
    es.indices.put_settings(index="logstash",
                            body={"index": {
                                "max_result_window": 50000
                            }})

    # 검색 쿼리
    res = es.search(index='logstash',
                    body={"query": {"match": {"id": user}},
                          "sort": [{"@timestamp": "desc"}]},
                    size=1000
                    )
    res_list = [el['_source']['clicked_repairshop_id'] for el in res['hits']['hits']]
    new_list = [i for i in res_list if i != None]
    result = []
    i = 0
    while i < 3 and i < len(new_list):
        search = es.search(index='repairshop',
                        body={"query": {"match": {"s_id": new_list[i]}}})
        result.append(search['hits']['hits'][0]['_source'])
        i+=1
    return JsonResponse({"list": result})


def side03(request):
    return render(request, 'side03.html')

def elasticsearch(request):
    # 검색어, 영업상태, 정비소구분 get
    words = request.GET.get('words')
    s_type = request.GET.get('s_type')
    running = request.GET.get('running')
    print(words)

    # logging
    # id, main_function, client_ip, searched_word
    logging_text = request.user.username + ',' + 'search_repairshop' + ',' + request.META.get('REMOTE_ADDR') + ',' + words
    logging(logging_text)

    # 입력값이 없을 때
    if words == '':
        words = '*'
    if s_type == '':
        s_type = '*'
    if running == '':
        running = '*'

    # elasticsearch 연결
    end_point = ELASTIC_CONFIG['url']
    print(end_point)
    es = Elasticsearch(hosts=end_point)

    # 검색할 index
    index_name = 'repairshop'

    # 검색 결과 수 제한 수정
    es.indices.put_settings(index="repairshop",
                            body={"index": {
                                "max_result_window": 50000
                            }})

    # 검색 쿼리
    res = es.search(index=index_name,
                    query={
                        "query_string": {
                            "query":
                                "(s_type: " + s_type + ") AND (running: " + running +
                                ") AND (s_name: " + words + " OR address:" + words + ")"
                        }
                    },
                    size=50000)
    result_list = [el['_source'] for el in res['hits']['hits']]

    for i in range(len(result_list)):
        result_list[i]['id'] = i + 1
        result_list[i]["time"] = result_list[i]["open_time"] + " - " + result_list[i]["close_time"]

    return JsonResponse({"list": result_list})

def admin(request):
    return render(request, 'admin.html', {'list':User.objects.all()})

def main01(request):
    # logging
    # id, main_function, client_ip, searched_word
    logging_text = request.user.username + ',' + 'show_cost' + ',' + request.META.get('REMOTE_ADDR') + ','
    logging(logging_text)

    res_data = dict()
    car_names_csv = pd.read_csv('./model/model/label_data.csv')
    car_names_df = pd.DataFrame(car_names_csv)
    car_names = car_names_df['car_name']
    car_names_list = car_names.values.tolist()
    res_data['car_name_list'] = car_names_list
    if request.method == 'POST':
        file = request.FILES.get('chooseFile')
        if file:
            image = Image(file=file)
            image.save()

            # 1
            class Unet(nn.Module):
                def __init__(self, num_classes, encoder, pre_weight):
                    super().__init__()
                    self.model = smp.Unet(classes=num_classes,
                                          encoder_name=encoder,
                                          encoder_weights=pre_weight,
                                          in_channels=3)

                def forward(self, x):
                    y = self.model(x)
                    encoder_weights = "imagenet"
                    return y

            # 2
            labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
            models = []

            n_classes = 2
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            for label in labels:
                model_path = f'./model/models/[DAMAGE][{label}]Unet.pt'

                model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
                model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                model.eval()

                models.append(model)

            # 3
            # 테스트 이미지 경로
            path_test = './media/'
            test_img = Image.objects.all().last()
            test_img = str(test_img)

            # 4
            ## 테스트 이미지에 id와 car_name 값 입력
            max_id = 18246
            test_car_name = ['']
            car_name = request.POST['car_name']
            repair_cost = {
                'Breakage_3': [],
                'Crushed_2': [],
                'Scratch_0': [],
                'Seperated_1': [],
                'car_name': [car_name],
                'id': [],
            }

            repair_cost['id'].append(max_id)

            # 5
            for img_list in range(1):
                img = cv2.imread(path_test + test_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))

                img_input = img / 255
                img_input = img_input.transpose([2, 0, 1])
                img_input = torch.tensor(img_input).float().to(device)
                img_input = img_input.unsqueeze(0)

                outputs = []
                for i, model in enumerate(models):
                    output = model(img_input)

                    img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
                    img_output = img_output.transpose([1, 2, 0])

                    outputs.append(img_output)

                for i, label in enumerate(labels):
                    repair_cost[label].append(outputs[i].sum())

            # 6
            model_path = './model/model/XGB03.model'
            new_model = joblib.load(model_path)

            # 7
            df_rp = pd.DataFrame(repair_cost)

            # 8
            label_info = pd.read_csv('./model/model/label_data.csv')
            for i in range(len(label_info['car_name'])):
                if df_rp['car_name'][0] == label_info['car_name'][i]:
                    df_rp = df_rp.replace(label_info['car_name'][i], label_info['labeled'][i])

            result = new_model.predict(df_rp.iloc[0].values.reshape(1, 6))
            result = format(round(result[0]), ',')

            res_data['status'] = 'ok'
            res_data['result'] = result
            res_data['car_name'] = repair_cost['car_name']

        return render(request, 'main01.html', res_data)
    return render(request, 'main01.html', res_data)

def visitor(request):
    return render(request, 'visitor.html')

def usage(request):
    return render(request, 'usage.html')

def user(request):
    return render(request, 'user.html', {'list' : User.objects.all()})

#####################################################################
#       과실비율 과실비율 과실비율 과실비율 과실비율 과실비율 과실비율 과실비율       #
#####################################################################

static_dir = '/home/godjy52/project/case04/django/static/'

def codeA(request):
    main_code = request.body.decode('utf-8')[-1]

    def make_resdict1(case):
        res_dict = {}
        jsonfile = static_dir + case + '_code.json'
        csvfile = static_dir + case + '_text.csv'
        with open(jsonfile, 'r') as jf:
            jsondata = json.load(jf)
            keys = jsondata.keys()
            with open(csvfile, 'r') as cf:
                reader = csv.reader(cf)
                for row in reader:
                    if row[0] in keys:
                        res_dict[row[0]] = row[1]
                        continue
        return res_dict
            

    if main_code == '1':
        case = 'cvc'
        res_dict = make_resdict1(case)
        return JsonResponse(res_dict)
      
    elif main_code == '2':
        case = 'highway'
        res_dict = make_resdict1(case)
        return JsonResponse(res_dict)

    elif main_code == '3':
        case = 'cvm'
        res_dict = make_resdict1(case)
        return JsonResponse(res_dict)

    elif main_code == '4':
        case = 'cva'
        res_dict = make_resdict1(case)
        return JsonResponse(res_dict)

    else:
        case = 'cvb'
        res_dict = make_resdict1(case)
        return JsonResponse(res_dict)


def codeB(request):
    data = request.body.decode('utf-8')
    main_code = data.split('&')[0][-1]
    code_A = data.split('&')[1][7:]

    def make_resdict2(case):
        res_dict = {}
        jsonfile = static_dir + case + '_code.json'
        csvfile = static_dir + case + '_text.csv'
        with open(jsonfile) as jf:
            jsondata = json.load(jf)
            for i in jsondata[code_A]:
                code_B_list = i.keys()
            with open(csvfile, 'r') as cf:
                reader = csv.reader(cf)
                for row in reader:
                    if row[0] in code_B_list:
                        res_dict[row[0]] = row[1]
                        continue
        return res_dict

    if main_code == '1':
        case = 'cvc'
        res_dict = make_resdict2(case)
        return JsonResponse(res_dict)

    elif main_code == '2':
        case = 'highway'
        res_dict = make_resdict2(case)
        return JsonResponse(res_dict)

    elif main_code == '3':
        case = 'cvm'
        res_dict = make_resdict2(case)
        return JsonResponse(res_dict)

    elif main_code == '4':
        case = 'cva'
        res_dict = make_resdict2(case)
        return JsonResponse(res_dict)

    else:
        case = 'cvb'
        res_dict = make_resdict2(case)
        return JsonResponse(res_dict)
    
def codeC(request):
    data = request.body.decode('utf-8')
    main_code = data.split('&')[0][-1]
    code_A = data.split('&')[1][7:]
    code_B = data.split('&')[2][7:]

    def make_resdict3(case):
        res_dict = {}
        jsonfile = static_dir + case + '_code.json'
        csvfile = static_dir + case + '_text.csv'
        with open(jsonfile) as jf:
            jsondata = json.load(jf)
            for i in jsondata[code_A][0][code_B]:
                code_C_list = i.keys()
            with open(csvfile, 'r') as cf:
                reader = csv.reader(cf)
                for row in reader:
                    if row[0] in code_C_list:
                        res_dict[row[0]] = row[1]
                        continue
        return res_dict

    if main_code == '1':
        case = 'cvc'
        res_dict = make_resdict3(case)
        return JsonResponse(res_dict)

    elif main_code == '2':
        case = 'highway'
        res_dict = make_resdict3(case)
        return JsonResponse(res_dict)

    elif main_code == '3':
        case = 'cvm'
        res_dict = make_resdict3(case)
        return JsonResponse(res_dict)

    elif main_code == '4':
        case = 'cva'
        res_dict = make_resdict3(case)
        return JsonResponse(res_dict)

    else:
        case = 'cvb'
        res_dict = make_resdict3(case)
        return JsonResponse(res_dict)

def codeD(request):
    data = request.body.decode('utf-8')
    main_code = data.split('&')[0][-1]
    code_A = data.split('&')[1][7:]
    code_B = data.split('&')[2][7:]
    code_C = data.split('&')[3][7:]

    def make_resdict4(case):
        res_dict = {}
        jsonfile = static_dir + case + '_code.json'
        csvfile = static_dir + case + '_text.csv'
        with open(jsonfile) as jf:
            jsondata = json.load(jf)
            for i in jsondata[code_A][0][code_B][0][code_C]:
                code_D_list = i.keys()
                with open(csvfile, 'r') as cf:
                    reader = csv.reader(cf)
                    for row in reader:
                        if row[0] in code_D_list:
                            res_dict[row[0]] = row[1]
                            continue
        return res_dict

    if main_code == '1':
        case = 'cvc'
        res_dict = make_resdict4(case)
        return JsonResponse(res_dict)

    elif main_code == '2':
        case = 'highway'
        res_dict = make_resdict4(case)
        return JsonResponse(res_dict)

    elif main_code == '3':
        case = 'cvm'
        res_dict = make_resdict4(case)
        return JsonResponse(res_dict)

    elif main_code == '4':
        case = 'cva'
        res_dict = make_resdict4(case)
        return JsonResponse(res_dict)

    else:
        case = 'cvb'
        res_dict = make_resdict4(case)
        return JsonResponse(res_dict)

def rate(request):
    # logging
    # id, main_function, client_ip, searched_word
    logging_text = request.user.username + ',' + 'acc_rate' + ',' + request.META.get('REMOTE_ADDR') + ','
    logging(logging_text)
    data = request.body.decode('utf-8')
    main_code = data.split('&')[0][-1]
    code_A = data.split('&')[1][7:]
    code_B = data.split('&')[2][7:]
    code_C = data.split('&')[3][7:]
    code_D = data.split('&')[4][7:]

    def make_rate(case):
        jsonfile = static_dir + case + '_code.json'
        with open(jsonfile) as jf:
            jsondata = json.load(jf)
            rate_list = jsondata[code_A][0][code_B][0][code_C][0][code_D]
        return rate_list[0]

    if main_code == '1':
        case = 'cvc'
        rate = make_rate(case)
        return HttpResponse(rate)

    elif main_code == '2':
        case = 'highway'
        rate = make_rate(case)
        return HttpResponse(rate)

    elif main_code == '3':
        case = 'cvm'
        rate = make_rate(case)
        return HttpResponse(rate)

    elif main_code == '4':
        case = 'cva'
        rate = make_rate(case)
        return HttpResponse(rate)

    else:
        case = 'cvb'
        rate = make_rate(case)
        return HttpResponse(rate)
