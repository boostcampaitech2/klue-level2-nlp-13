from inference import do_inference
import wandb
import sys, getopt
import pandas as pd

from load_data import *
from models import *
from mytokenizers import *
from train import *
from utils import *

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    argv = sys.argv
    file_name = argv[0] # 실행시키는 파일명
    config_path = ""   # config file 경로
    
    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "config_path="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(file_name, "-c <config_path>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-c <config_path>")
            sys.exit(0)
        elif opt in ("-c", "--config_path"):
            config_path = arg
    
    # 입력이 필수적인 옵션 입력이 없으면 오류 메시지 출력
    if len(config_path) < 1:
        print(file_name, "-c <config_path> is madatory")
        sys.exit(2)
    
    # 1. Load config file
    print('======== Config loading...'.ljust(50, "="))
    config = read_config(config_path)
    print('======== END '.ljust(50, "="))
    
    seed_everything(config.seed)

    # 2. Load data
    print('======== Data loading...'.ljust(50, "="))
    dataset = load_data(config.data_path)
    
    # 2.1 이미 만들어둔 dataset을 사용
    #dataset = pd.read_csv('../../dataset/train/train_dataset.csv')

    # class imbalanced 보완을 위한 loss 사용시, 가중치 계산 (utills.py)
    if 'weighted' in  config.loss_name:
        label = label_to_num(config, dataset['label'].values)
        config.class_weight = get_class_weights(label)
    
    # 데이터 라벨의 비율 맞게 훈련/검증 데이터 분리
    #train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], random_state=config.random_state)
    dev_json_path = "../../dataset/train/klue-re.json"
    train_dataset = dataset
    valid_dataset = preprocessing_dataset(json_to_df(dev_json_path))

    print(train_dataset.head())
    print(valid_dataset.head())

    # Augmentation data 추가
    #additional_dataset = pd.read_csv('../../dataset/train/additional_dataset.csv')
    #train_dataset = pd.concat([train_dataset, additional_dataset], ignore_index=True)

    # 라벨-index 맵핑
    train_label = label_to_num(config, train_dataset['label'].values)
    valid_label = label_to_num(config, valid_dataset['label'].values)
    print('======== END '.ljust(50, "="))

    # 3. Tokenizing
    print('='*10, "Tokenizing...", '='*10)
    # 토크나이저 불러오기 (tokenizer.py)
    tokenizer = get_tokenizer(config.tokenizer_name)
    # 훈련/검증 각각 tokenizer 적용
    if config.add_special_token:
        print("Add special token")
        # 추가 하고 싶은 Special token dict 정의
        special_tokens_dict = {'additional_special_tokens': config.new_special_token_list}
        # tokenizer에 더해주기
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenized_train = tokenized_dataset(config, train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(config, valid_dataset, tokenizer)
    print(tokenizer)
    print('='*10, "END", '='*10)

    # 4. Make pytorch dataset
    print('='*10, "Make pytorch dataset...", '='*10)
    # 각각 데이터셋 클래스 적용
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)
    print('='*10, "END", '='*10)

    # 5. Make model
    print('='*10, "Make model...", '='*10)
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    MODEL_NAME = './pre_model'
    #model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(config.device)

    # 모델 불러오기 (models.py)
    #model = get_model(config.model_name, config.num_classes).to(config.device)
    model = get_model(config.model_name, config.num_classes).to(config.device)
    # 모델도 늘려줘야함!
    if config.add_special_token:
        model.resize_token_embeddings(len(tokenizer))
    
    print('='*10, "END", '='*10)

    # 6. Train
    print('='*10, "Start train...", '='*10)
    print(f'RE_train_dataset : {len(RE_train_dataset)}')
    wandb.init(project=config.project, entity=config.entity, name=config.run_name, config=config)
    train(config, model, RE_train_dataset, RE_valid_dataset)
    print('='*10, "END", '='*10)

    # 7. Inference
    if config.make_inference:
        print('='*10, "Start inference...", '='*10)
        do_inference(config)

    print('='*20)
    print('job done!')