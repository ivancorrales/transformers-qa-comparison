import os
import json
import requests


DATASETS_DIR = f'.datasets'
SQUAD_ES_v2_FILENAME = 'dev-v2.0-es_small.json'
SQUAD_ES_v2_URL = f'https://github.com/ccasimiro88/TranslateAlignRetrieve/raw/master/SQuAD-es-v2.0/{SQUAD_ES_v2_FILENAME}'
SQUAD_ES_v2_PATH = f'{DATASETS_DIR}/{SQUAD_ES_v2_FILENAME}'

class SquadV2:

    def __init__(self):
        self.__download_dataset()
        self.__read_dataset_from_json()
        pass
    
    def __download_dataset(self):
        if not os.path.exists(SQUAD_ES_v2_PATH):
            print('Downloading SQuAD-es-v2.0...')
            os.makedirs(DATASETS_DIR, exist_ok=True)
            res = requests.get(SQUAD_ES_v2_URL)
            with open(SQUAD_ES_v2_PATH,'wb') as f:
                for chunk in res.iter_content(chunk_size=4):
                    f.write(chunk)
        else:
            print('Dataset SQuAD-es-v2.0 is already downloaded')

    def __read_dataset_from_json(self, max_start=512):
        contexts = []
        questions = []
        answers = []
        with open(SQUAD_ES_v2_PATH, 'rb') as f:
            squad_dict = json.load(f)

        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    access = 'plausible_answers' if 'plausible_answers' in qa.keys() else 'answers'
                    for answer in qa[access]:
                        if answer['answer_start'] > -1 and answer['answer_start'] < max_start:  
                            contexts.append(context)
                            questions.append(question)
                            answers.append(answer)
        
        train_len     = int(len(contexts)*.8)
        # trainning data
        self.__train_contexts  = contexts[:train_len]
        self.__train_questions = questions[:train_len]
        self.__train_answers   = answers[:train_len]
        
        # validation data
        self.__val_contexts  = contexts[:-train_len]
        self.__val_questions = questions[:-train_len]
        self.__val_answers   = answers[:-train_len]


    def train_contexts(self):
        return self.__train_contexts
    
    def train_questions(self):
        return self.__train_questions
    
    def train_answers(self):
        return self.__train_answers

    def val_contexts(self):
        return self.__val_contexts
    
    def val_questions(self):
        return self.__val_questions
    
    def val_answers(self):
        return self.__val_answers
        


if __name__ == '__main__':
    SquadV2()