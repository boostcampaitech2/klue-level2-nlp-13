import os
import sys, getopt

from load_data import *
from models import *
from mytokenizers import *
from train import *
from utills import *

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
    
    # 입력이 필수적인 옵션
    if len(config_path) < 1:
        print(file_name, "-c <config_path> is madatory")
        sys.exit(2)
    
    # 1. Load config file
    print('='*10, "Config loading...", '='*10)
    config = read_config(config_path)
    print('='*10, "END", '='*10)
    
    # 2. Load data
    print('='*10, "Data loading...", '='*10)
    train_dataset = load_data(config.data_path)
    train_label = label_to_num(train_dataset['label'].values)
    print('='*10, "END", '='*10)

    # 3. Tokenizing
    print('='*10, "Tokenizing...", '='*10)
    tokenizer = get_tokenizer(config.tokenizer_name)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    print('='*10, "END", '='*10)

    # 4. Make pytorch dataset
    print('='*10, "Make pytorch dataset...", '='*10)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    print('='*10, "END", '='*10)

    # 5. Make model
    print('='*10, "Make model...", '='*10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.model_name, config.num_classes)
    print('='*10, "END", '='*10)

    # 6. Train
    print('='*10, "Start train...", '='*10)
    train(config, model, RE_train_dataset)
    print('='*10, "END", '='*10)