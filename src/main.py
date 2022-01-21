
from importlib.abc import SourceLoader
import torch

from transformers import BertTokenizerFast
from models.dataset import Dataset
from models.trainer import Trainner
from sources.source import SourceInterface
from sources.squad import SquadSource

BERT_MODEL_NAME = 'mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'



if __name__ == '__main__':
    adapter: SourceInterface = SquadSource('..')
    print(f'Loading tokenizer {BERT_MODEL_NAME}')
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    train_encodings = adapter.train_tokenizer(tokenizer)
    val_encodings = adapter.val_tokenizer(tokenizer)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainner = Trainner(BERT_MODEL_NAME, train_encodings,device)
    trainner.train()
    
