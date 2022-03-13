# Created by Hansi at 1/7/2022
import logging
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold, StratifiedShuffleSplit

from src.models.lstm_model import NNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, 'output')

SEED = 42

config = {
    'manual_seed': SEED,
    'best_model_dir': os.path.join(OUTPUT_DIRECTORY, "model"),

    'max_len': 128,  # max sequence length
    'max_features': None,  # how many unique words to use (i.e num rows in embedding vector)
    'num_train_epochs': 20,

    'train_batch_size': 32,
    'test_batch_size': 32,

    'early_stopping': True,
    'early_stopping_patience': 5,  # 2

    'learning_rate': 1e-3,

    'train_file': 'train.tsv',  # filename to save train data
    'dev_file': 'dev.tsv',  # filename to save dev data
    'dev_size': None,  # 0.1

    'labels_list': [0, 1],
    'embedding_details': {'fasttext': '/content/crawl-300d-2M-subword/crawl-300d-2M-subword.vec'},
}


def cross_validate(train_file_path, k_folds, config):
    data = pd.read_csv(train_file_path, sep=",", encoding="utf-8")
    data = data[['text', 'label']]
    data['text'] = data['text'].apply(lambda x: x.lower())

    dict_results = dict()
    kf = KFold(n_splits=k_folds, random_state=config['manual_seed'], shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        # delete create folder
        if os.path.exists(OUTPUT_DIRECTORY):
            shutil.rmtree(OUTPUT_DIRECTORY)
        os.makedirs(OUTPUT_DIRECTORY)

        new_data_dir = os.path.join(OUTPUT_DIRECTORY, f"data")
        os.makedirs(new_data_dir)

        logger.info(f"FOLD: {fold + 1}, TRAIN: {train_index}, TEST: {test_index}")
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        if config['dev_size'] is not None:
            train, dev = split_data(data, SEED, test_size=config['dev_size'])

        else:
            dev = test

        train.to_csv(os.path.join(new_data_dir, config['train_file']), sep="\t", index=False)
        logger.info(f"Saved {train.shape[0]} train instances.")
        dev.to_csv(os.path.join(new_data_dir, config['dev_file']), sep="\t", index=False)
        logger.info(f"Saved {dev.shape[0]} dev instances.")

        # train model
        logger.info(f"Training model...")
        model = NNModel('lstm', data_dir=new_data_dir, args=config)
        model.train()

        # evaluate model
        if config['dev_size'] is not None:
            # get model predictions
            preds, raw_preds = model.predict(test['text'].tolist())
        else:
            preds, raw_preds = model.predict(dev['text'].tolist())

        eval_results = get_eval_results(test['label'].tolist(), preds)
        logger.info(f'{fold + 1} test results: {eval_results}')
        dict_results[fold + 1] = eval_results

    # calculate average results
    # average_recall = np.asarray([d['recall'] for d in dict_results.values()]).mean()
    logger.info(f"average_recall: {np.asarray([d['recall'] for d in dict_results.values()]).mean()}")
    logger.info(f"average_precision: {np.asarray([d['precision'] for d in dict_results.values()]).mean()}")
    logger.info(f"average_f1: {np.asarray([d['f1'] for d in dict_results.values()]).mean()}")
    logger.info(f"average_accuracy: {np.asarray([d['accuracy'] for d in dict_results.values()]).mean()}")
    logger.info(f"average_mcc: {np.asarray([d['mcc'] for d in dict_results.values()]).mean()}")


def train(train_file_path, config, test_file_path=None, evaluate=True):
    data = pd.read_csv(train_file_path, sep=",", encoding="utf-8")
    data = data[['index', 'text', 'label']]
    data['text'] = data['text'].apply(lambda x: x.lower())

    if test_file_path:
        test = pd.read_csv(test_file_path, sep=",", encoding="utf-8")
        test = test[['index', 'text', 'label']]
        test['text'] = test['text'].apply(lambda x: x.lower())

    # delete create folder
    if os.path.exists(OUTPUT_DIRECTORY):
        shutil.rmtree(OUTPUT_DIRECTORY)
    os.makedirs(OUTPUT_DIRECTORY)

    new_data_dir = os.path.join(OUTPUT_DIRECTORY, f"data")
    os.makedirs(new_data_dir)

    if config['dev_size'] is not None:
        train, dev = split_data(data, SEED, test_size=config['dev_size'])
    else:
        if test_file_path is None:
            raise ValueError("No dev size or test file path is provided!")
        train = data
        dev = test

    train.to_csv(os.path.join(new_data_dir, config['train_file']), sep="\t", index=False)
    logger.info(f"Saved {train.shape[0]} train instances.")
    dev.to_csv(os.path.join(new_data_dir, config['dev_file']), sep="\t", index=False)
    logger.info(f"Saved {dev.shape[0]} dev instances.")

    # train model
    logger.info(f"Training model...")
    model = NNModel('lstm', data_dir=new_data_dir, args=config)
    model.train()

    # predictions
    if test_file_path is not None:
        preds, raw_preds = model.predict(test['text'].tolist())
        test['predictions'] = preds
        save_predictions(test, os.path.join(OUTPUT_DIRECTORY, "submission.json"))

        # evaluate
        if evaluate:
            eval_results = get_eval_results(test['label'].tolist(), preds)
            logger.info(f'Test results: {eval_results}')


def split_data(df, seed, label_column='label', test_size=0.1):
    y = df[label_column]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(sss.split(df, y))

    train = df.iloc[train_index]
    test = df.iloc[test_index]
    return train, test


def get_eval_results(actuals, predictions):
    results = dict()
    r = recall_score(actuals, predictions)
    results['recall'] = r
    p = precision_score(actuals, predictions)
    results['precision'] = p
    f1 = f1_score(actuals, predictions)
    results['f1'] = f1
    accuracy = accuracy_score(actuals, predictions)
    results['accuracy'] = accuracy
    mcc = matthews_corrcoef(actuals, predictions)
    results['mcc'] = mcc
    return results


def save_predictions(test_data, submission_file_path):
    with open(submission_file_path, 'w') as f:
        for index, row in test_data.iterrows():
            item = {"index": row['index'], "prediction": row['predictions']}
            f.write("%s\n" % item)


if __name__ == '__main__':
    # train_file_path = os.path.join(BASE_PATH, 'data/all.csv')
    # k_folds = 5
    # cross_validate(train_file_path, k_folds, config)

    train_file_path = os.path.join(BASE_PATH, 'data/CTB_forCASE.csv')
    test_file_path = os.path.join(BASE_PATH, 'data/all.csv')
    train(train_file_path, config, test_file_path=test_file_path)
