
MAX_LEN = 512

class SourceInterface:

    def train_contexts(self) -> list:
        pass
    
    def train_questions(self) -> list:
        pass
    
    def train_answers(self) -> list:
        pass

    def val_contexts(self) -> list:
        pass
    
    def val_questions(self) -> list:
        pass
    
    def val_answers(self) -> list:
        pass

    def train_tokenizer(self,tokenizer, truncation=True, padding=True, max_len =MAX_LEN):
        encodings =  tokenizer(
            self.train_contexts(), 
            self.train_questions(), 
            truncation=truncation, 
            padding=padding, 
            max_length=max_len,
        )
        self.__add_token_positions(encodings, self.train_answers())
        return encodings
    
    def val_tokenizer(self,tokenizer, truncation=True, padding=True, max_len =MAX_LEN):
        encodings =  tokenizer(
            self.val_contexts(), 
            self.val_questions(), 
            truncation=truncation, 
            padding=padding, 
            max_length=max_len,
        )
        self.__add_token_positions(encodings, self.val_answers())
        return encodings
    
    def __add_token_positions(self, encodings,answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i,answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i,answers[i]['answer_end']))
            if start_positions[-1] is None:
                start_positions[-1] = MAX_LEN
            shift=1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i,answers[i]['answer_end'] - shift)
                shift += 1
        encodings.update({
            'start_positions': start_positions,
            'end_positions': end_positions,
        })