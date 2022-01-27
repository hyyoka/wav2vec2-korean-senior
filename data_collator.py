from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import ast
from transformers import Wav2Vec2Processor
from torch.nn.utils.rnn import pad_sequence

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor.from_pretrained("hyyoka/wav2vec2-xlsr-korean-senior", pad_token_id=49)
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def mapping(self,data):
        with self.processor.as_target_processor():
            ret = self.processor("".join([i if i!='\x1b' else '|' for i in ast.literal_eval(data)])).input_ids
            ret_torch = torch.tensor([int(0 if value is None else value) for value in ret])
        return ret_torch

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature['input_values']} for feature in features]
        # e.g. feature['label'] = "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        label_features = [{"input_ids": self.mapping(feature["label"])} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


# 자소 단위로 나누어지지 않은 경우 사용 -> 한번 하면 저장해놓기
class DataProc:
    def __init__(self, model_name="hyyoka/wav2vec2-xlsr-korean-senior"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, pad_token_id=49)

    def to_jaso(self, sentence):
        NO_JONGSUNG = ''
        CHOSUNGS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        JOONGSUNGS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        JONGSUNGS = [NO_JONGSUNG,  'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        N_CHOSUNGS, N_JOONGSUNGS, N_JONGSUNGS = 19, 21, 28
        FIRST_HANGUL, LAST_HANGUL = 0xAC00, 0xD7A3 #'가', '힣'    
     
        result = []
        for char in sentence:
            if ord(char) < FIRST_HANGUL or ord(char) > LAST_HANGUL: 
                result.append('|')
            else:          
                code = ord(char) - FIRST_HANGUL
                jongsung_index = code % N_JONGSUNGS
                code //= N_JONGSUNGS
                joongsung_index = code % N_JOONGSUNGS
                code //= N_JOONGSUNGS
                chosung_index = code
                result.append(CHOSUNGS[chosung_index])
                result.append(JOONGSUNGS[joongsung_index])
                if jongsung_index!=0:
                    result.append(JONGSUNGS[jongsung_index])
                
        with self.processor.as_target_processor():
            ret = self.processor("".join(result))
        
        return ret.input_ids

    def prepare_dataset(self, df):
        """
        df.cols = ['audio', 'sentence', 'path']
        """
        df['label'] = df['sentence'].apply(self.to_jaso)
        return df