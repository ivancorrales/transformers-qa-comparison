from tqdm import tqdm
from transformers import AdamW, BertForQuestionAnswering
from torch.utils.data import DataLoader

from models.dataset import Dataset

class Trainner():

    def __init__(self, model,encodings, device) -> None:
        self.bert = BertForQuestionAnswering.from_pretrained(model)
        self.bert.to(device)
        self.device = device
        self.train_dataset = Dataset(encodings)  
        
    def train(self,n_epochs = 1, batch_size=16, lr=5e-5 ):
        print('Trainning model...')
        self.bert.train()
        optim = AdamW(self.bert.parameters(), lr=lr)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(n_epochs):
            loop = tqdm(train_loader)
            for batch in loop:
                optim.zero_grad()
      
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)

                outputs = self.bert(input_ids, 
                      attention_mask=attention_mask, 
                      start_positions=start_positions,
                      end_positions=end_positions)
      
                loss = outputs[0]
                loss.backward()
                optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
