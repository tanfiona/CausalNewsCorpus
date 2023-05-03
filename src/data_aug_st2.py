from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
from copy import copy
import re

NUM_RETURN_SEQ = 1


class PhraseGenerator(object):
    def __init__(self):
        model_name = 'tuner007/pegasus_paraphrase'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        model.cuda()
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def get_response(self, input_text, num_return_sequences=1, num_beams=20):
        batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to("cuda")
        translated = self.model.generate(
            **batch,
            max_length=60,
            no_repeat_ngram_size=4,
            encoder_no_repeat_ngram_size=2,
            repetition_penalty=1.0,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences, 
            temperature=1.5
        )
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def data_augmentation_paraphrase(self, text):
        arg1 = text.split("</ARG1>")[0].split("<ARG1>")[1]
        paraphrase_arg1 = self.get_response(arg1, num_return_sequences=NUM_RETURN_SEQ)
        paraphrase_arg1 = [_[:-1] if _.endswith('.') and not arg1.endswith('.') else _ for _ in paraphrase_arg1]
        paraphrase_arg1 = [_[0].lower() + _[1:] if _[0].isupper() and not arg1[0].isupper() else _ for _ in paraphrase_arg1]
        paraphrase_arg1 = [_ for _ in paraphrase_arg1 if _ != arg1]

        arg0 = text.split("</ARG0>")[0].split("<ARG0>")[1]
        paraphrase_arg0 = self.get_response(arg0, num_return_sequences=NUM_RETURN_SEQ)
        paraphrase_arg0 = [_[:-1] if _.endswith('.') and not arg0.endswith('.') else _ for _ in paraphrase_arg0]
        paraphrase_arg0 = [_[0].lower() + _[1:] if _[0].isupper() and not arg0[0].isupper() else _ for _ in paraphrase_arg0]
        paraphrase_arg0 = [_ for _ in paraphrase_arg0 if _ != arg0]
        
        # remove signals that overlap into cause-effect args
        p_arg1 = p_arg0 = 'dummy holding text'
        old_text = copy(text)
        text = re.sub("<ARG1>.*</ARG1>", f"<ARG1>{p_arg1}</ARG1>", text)
        text = re.sub("<ARG0>.*</ARG0>", f"<ARG0>{p_arg0}</ARG0>", text)
        original_signals = re.findall('</*SIG\d*>', old_text)
        final_signals = re.findall('</*SIG\d*>', text)
        if len(original_signals) != len(final_signals):
            missing_signals = [i for i in original_signals if i not in final_signals]
            for signal_bound in missing_signals:
                if '/' in signal_bound:
                    text = re.sub(re.sub('/','',signal_bound),'',text)
                else:
                    text = re.sub(signal_bound[0]+'/'+signal_bound[1:],'',text)

        augmented_text = list()
        for p_arg1 in paraphrase_arg1:
            for p_arg0 in paraphrase_arg0:
                tmp = re.sub("<ARG1>.*</ARG1>", f"<ARG1>{p_arg1}</ARG1>", text)
                tmp = re.sub("<ARG0>.*</ARG0>", f"<ARG0>{p_arg0}</ARG0>", tmp)
                augmented_text.append(tmp)
        
        return augmented_text


if __name__ == "__main__":
    generator = PhraseGenerator()
    
    output_data = {
        "causal_text_w_pairs": [],
        "corpus": [], 
        "doc_id": [],
        "sent_id": [],
        "eg_id": [],
        "index": [],
        "text": [],
        "num_rs": [],
    }
    data = pd.read_csv("data/V2/train_subtask2.csv")
    for text in tqdm(list(data["text_w_pairs"]), desc="Paraphrasing"):
        for key in output_data.keys():
            if key == "causal_text_w_pairs":
                output_data[key].append(generator.data_augmentation_paraphrase(text))
            else:
                output_data[key].append(0)
    pd.DataFrame(output_data).to_csv(f"data/V2/augmented_subtask2_{NUM_RETURN_SEQ**2}_train.csv")