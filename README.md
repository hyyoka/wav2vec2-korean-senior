# wav2vec2-korean-senior

Futher fine-tuned [fleek/wav2vec-large-xlsr-korean](https://huggingface.co/fleek/wav2vec-large-xlsr-korean) using the [AIhub 자유대화 음성(노인남녀)](https://aihub.or.kr/aidata/30704).

- Total train data size: 808,642
- Total vaild data size: 159,970

When using this model, make sure that your speech input is sampled at 16kHz.

### Inference

```py
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import re

def clean_up(transcription):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', transcription)
    return result

model_name "hyyoka/wav2vec2-xlsr-korean-senior"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
speech_array, sampling_rate = torchaudio.load(wav_file)
feat = processor(speech_array[0], 
                            sampling_rate=16000, 
                            padding=True,
                            max_length=800000, 
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors="pt",
                            pad_token_id=49
                            )
input = {'input_values': feat['input_values'],'attention_mask':feat['attention_mask']}

outputs = model(**input, output_attentions=True)
logits = outputs.logits
predicted_ids = logits.argmax(axis=-1)
transcription = processor.decode(predicted_ids[0])
stt_result = clean_up(transcription)


```
