import os
import json
import requests

from sources.source import SourceInterface


DATASETS_DIR = f'.datasets'
SQUAD_ES_v2_FILENAME = 'dev-v2.0-es_small.json'
SQUAD_ES_v2_URL = f'https://github.com/ccasimiro88/TranslateAlignRetrieve/raw/master/SQuAD-es-v2.0/{SQUAD_ES_v2_FILENAME}'


class SquadSource(SourceInterface):

    def __init__(self, root):
        self.datasets_path = os.path.join(root,DATASETS_DIR)
        self.__download_dataset()
        self.__read_dataset_from_json()
        pass
    
    def __download_dataset(self):
        dataset_filepath = f'{self.datasets_path}/{SQUAD_ES_v2_FILENAME}'
        if not os.path.exists(dataset_filepath):
            print('Downloading SQuAD-es-v2.0...')
            os.makedirs(self.datasets_path, exist_ok=True)
            res = requests.get(SQUAD_ES_v2_URL)
            with open(dataset_filepath,'wb') as f:
                for chunk in res.iter_content(chunk_size=4):
                    f.write(chunk)
        else:
            print('Dataset SQuAD-es-v2.0 is already downloaded')

    def __read_dataset_from_json(self, max_start=512):
        contexts = []
        questions = []
        answers = []
        dataset_filepath = f'{self.datasets_path}/{SQUAD_ES_v2_FILENAME}'
        with open(dataset_filepath, 'rb') as f:
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
        
        self.__process_data(contexts,questions, answers)
        

        train_len     = int(len(contexts)*.8)
        # trainning data
        self.__train_contexts  = contexts[:train_len]
        self.__train_questions = questions[:train_len]
        self.__train_answers   = answers[:train_len]
        
        # validation data
        self.__val_contexts  = contexts[:-train_len]
        self.__val_questions = questions[:-train_len]
        self.__val_answers   = answers[:-train_len]

    def __process_data(self, contexts,questions, answers):
        invalid_indexes = []
        for index,(answer, context) in enumerate(zip(answers, contexts)):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            else:
                modified = False
            for n in range(3):
                if context[start_idx+n:end_idx+n] == gold_text:
                    answer['answer_start'] = start_idx + n
                    answer['answer_end'] = end_idx + n
                    modified = True
            if not modified: 
                invalid_indexes.append(index)
        for index in reversed(invalid_indexes):
            del contexts[index]
            del questions[index]
            del answers[index]

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
        


