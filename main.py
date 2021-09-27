from inference import do_inference
import wandb
import sys, getopt

from load_data import *
from models import *
from mytokenizers import *
from train import *
from utills import *

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
    dataset = load_data(config.data_path)
        # class imbalanced 보완을 위한 loss 사용시, 가중치 계산 (utills.py)
    if 'weighted' in  config.loss_name:
        label = label_to_num(dataset['label'].values)
        config.class_weight = get_class_weights(label)
        # 데이터 라벨의 비율 맞게 훈련/검증 데이터 분리
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset['label'], random_state=config.random_state)
        # 라벨-index 맵핑
    train_label = label_to_num(train_dataset['label'].values)
    valid_label = label_to_num(valid_dataset['label'].values)
    print('='*10, "END", '='*10)

    # 3. Tokenizing
    print('='*10, "Tokenizing...", '='*10)
        # 토크나이저 불러오기 (tokenizer.py)
    tokenizer = get_tokenizer(config.tokenizer_name)
        # 훈련/검증 각각 tokenizer 적용
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)
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
        # 모델 불러오기 (models.py)
    model = get_model(config.model_name, config.num_classes).to(config.device)
    print('='*10, "END", '='*10)

    # 6. Train
    print('='*10, "Start train...", '='*10)
    wandb.init(project='LKH', entity='clue', name=config.run_name)
    train(config, model, RE_train_dataset, RE_valid_dataset)
    print('='*10, "END", '='*10)

    # 7. Inference
    print('='*10, "Start inference...", '='*10)
    do_inference(config) # 오류 있을 수 있음! 아직 여기까지 실행된 적이 없음!