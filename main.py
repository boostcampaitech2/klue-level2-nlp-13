#from inference import do_inference
import wandb
import sys, getopt

from load_data import *
from models import *
from mytokenizers import *
from train import *
from custom_train import *
from utills import *

import torch
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
    print('='*10, "Config loading...", '='*10)
    config = read_config(config_path)
    print('='*10, "END", '='*10)
    
    # 2. Load data
    print('='*10, "Data loading...", '='*10)
    train_dataset, valid_dataset = load_data(config.data_path, config, 'main')

        # class imbalanced 보완을 위한 loss 사용시, 가중치 계산 (utills.py)
    if 'weighted' in  config.loss_name:
        label = label_to_num(config, train_dataset['label'].values)
        config.class_weight = get_class_weights(label)
        # 데이터 라벨의 비율 맞게 훈련/검증 데이터 분리
    #train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], shuffle=True, random_state=config.random_state)
    ####################################################### temp 시도
    #no_rel_dataset = train_dataset.loc[train_dataset['label']=='no_relation']
    #train_dataset = train_dataset.loc[train_dataset['label']!='no_relation']
    #train_dataset = pd.concat([train_dataset, no_rel_dataset.iloc[:3000]], axis=0)
    ######################################################
    #print('loded:\t', train_dataset.shape)

        # 추가된 데이터를 사용 한다면
    if config.use_aug_data:
        aug_dataset = load_data(config.augmentation_data_path, config, 'aug')
        #aug_dataset = aug_dataset.loc[~aug_dataset['id'].isin(valid_dataset['id'])]
        print(train_dataset.shape, aug_dataset.shape)
        train_dataset = pd.concat([train_dataset, aug_dataset], axis=0)
        print('changed:\t', train_dataset.shape)
        # 라벨-index 맵핑
    train_label = label_to_num(config, train_dataset['label'].values)
    valid_label = label_to_num(config, valid_dataset['label'].values)
    print('='*10, "END", '='*10)

    # 3. Tokenizing
    print('='*10, "Tokenizing...", '='*10)
        # 토크나이저 불러오기 (tokenizer.py)
    tokenizer = get_tokenizer(config.tokenizer_name)
        # 훈련/검증 각각 tokenizer 적용
    if config.add_special_token == 'special':
        print("Add special token")
        # 추가 하고 싶은 Special token dict 정의
        special_tokens_dict = {'additional_special_tokens': config.new_special_token_list}
        # tokenizer에 더해주기
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenized_train = tokenized_dataset(config, train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(config, valid_dataset, tokenizer)
    
    if config.use_entity_embedding:
        insert_entity_idx_tokenized_dataset(tokenizer, tokenized_train, config)
        insert_entity_idx_tokenized_dataset(tokenizer, tokenized_valid, config)
    print('='*10, "END", '='*10)

    # 4. Make pytorch dataset
    print('='*10, "Make pytorch dataset...", '='*10)
        # 각각 데이터셋 클래스 적용
    RE_train_dataset = RE_Dataset(tokenized_train, train_label, config)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label, config)
    print('='*10, "END", '='*10)

    # 5. Make model
    print('='*10, "Make model...", '='*10)
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 모델 불러오기 (models.py)
    model = get_model(config.model_name, config.num_classes).to(config.device)
    # 모델도 늘려줘야함!
    #if config.add_special_token:
    model.resize_token_embeddings(len(tokenizer))
    print('='*10, "END", '='*10)

    # 6. Train
    print('='*10, "Start train...", '='*10)
    wandb.init(project=config.project, entity=config.entity, name=config.run_name, config=config)
    #train(config, model, RE_train_dataset, RE_valid_dataset)
    custom_train(config, model, RE_train_dataset, RE_valid_dataset, tokenizer)
    print('='*10, "END", '='*10)

    # 7. Inference
    # if config.prediction_mode not in ['binary', 'multi']:
    #     print('='*10, "Start inference...", '='*10)
    #     do_inference(config)