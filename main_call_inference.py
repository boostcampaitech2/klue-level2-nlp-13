from inference import do_inference
import wandb
import sys, getopt

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
    print('='*10, "Config loading...", '='*10)
    config = read_config(config_path)
    print('='*10, "END", '='*10)
    
    seed_everything(config.seed)

    # 7. Inference
    if config.make_inference:
        print('='*10, "Start inference...", '='*10)
        do_inference(config)

    print('='*20)
    print('job done!')