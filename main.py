##################
# import modules #
##################

import wandb
import sys, getopt
from load_data import *
from models import *
from mytokenizers import *
from train import *
from custom_train import *
from utills import *
import torch


########
# main #
########

if __name__ == "__main__":
    """
        Train Model for Korean Relation Extraction task
        python main.py -c <config file path>
        > python main.py -c configures/configure.ini
    """
    argv = sys.argv
    file_name = argv[0] # run file name
    config_path = ""    # config file path init
    
    try:
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "config_path="])
    except getopt.GetoptError:
        # Wrong option selected
        print(file_name, "-c <config_path>")
        sys.exit(2)
        
    # get inputs configure
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-c <config_path>")
            sys.exit(0)
        elif opt in ("-c", "--config_path"):
            config_path = arg
    
    # Check essential options
    if len(config_path) < 1:
        print(file_name, "-c <config_path> is madatory")
        sys.exit(2)
    
    #######################
    # 1. Load config file #
    #######################
    print('='*10, "Config loading...", '='*10)
    config = read_config(config_path)
    print('='*10, "END", '='*10)
    
    ################
    # 2. Load data #
    ################
    print('='*10, "Data loading...", '='*10)
    train_dataset, valid_dataset = load_data(config.data_path, config, 'main')

        # Loss Selection (utils.py)
    if 'weighted' in  config.loss_name:
        label = label_to_num(config, train_dataset['label'].values)
        config.class_weight = get_class_weights(label)

        # Additional Data
    if config.use_aug_data:
        aug_dataset = load_data(config.augmentation_data_path, config, 'aug')
        print(train_dataset.shape, aug_dataset.shape)
        train_dataset = pd.concat([train_dataset, aug_dataset], axis=0)
        print('changed:\t', train_dataset.shape)

        # Label-Index mapping
    train_label = label_to_num(config, train_dataset['label'].values)
    valid_label = label_to_num(config, valid_dataset['label'].values)
    print('='*10, "END", '='*10)

    #################
    # 3. Tokenizing #
    #################
    print('='*10, "Tokenizing...", '='*10)

        # Load tokenizer (tokenizer.py)
    tokenizer = get_tokenizer(config.tokenizer_name)

        # Train/Validation tokenizer
    if config.add_special_token == 'special':
        print("Add special token")
        # Define additional Special token dict
        special_tokens_dict = {'additional_special_tokens': config.new_special_token_list}
        # Add to tokenizer
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    tokenized_train = tokenized_dataset(config, train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(config, valid_dataset, tokenizer)
    
        # Use entity embedding
    if config.use_entity_embedding:
        insert_entity_idx_tokenized_dataset(tokenizer, tokenized_train, config)
        insert_entity_idx_tokenized_dataset(tokenizer, tokenized_valid, config)
    print('='*10, "END", '='*10)

    ###########################
    # 4. Make pytorch dataset #
    ###########################
    print('='*10, "Make pytorch dataset...", '='*10)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label, config)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label, config)
    print('='*10, "END", '='*10)

    #################
    # 5. Make model #
    #################
    print('='*10, "Make model...", '='*10)
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Load model (models.py)
    model = get_model(config.model_name, config.num_classes).to(config.device)
        # Model resize token embedding
    model.resize_token_embeddings(len(tokenizer))
    print('='*10, "END", '='*10)

    ############
    # 6. Train #
    ############
    print('='*10, "Start train...", '='*10)
    wandb.init(project=config.project, entity=config.entity, name=config.run_name, config=config)
        #setting mode for each train module
    if config.trainer:
        train(config, model, RE_train_dataset, RE_valid_dataset)
    else:
        custom_train(config, model, RE_train_dataset, RE_valid_dataset, tokenizer)
    print('='*10, "END", '='*10)