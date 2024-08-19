import torch
from utils import get_logger
from collections import defaultdict


def get_t5_featuremaps(featuremaps, encoder_hidden_states, decoder_hidden_states):
    enc_id_to_layer_name = {}
    enc_id_to_layer_name[0] = "encoder.init_embedding"
    for i in range(1, len(encoder_hidden_states)):
        enc_id_to_layer_name[i] = f"encoder.T5Block.{i-1}"
    dec_id_to_layer_name = {}
    dec_id_to_layer_name[0] = "decoder.init_embedding"
    for i in range(1, len(decoder_hidden_states[0])):
        dec_id_to_layer_name[i] = f"decoder.T5Block.{i-1}"
    
    for layer_id, layer_name in enc_id_to_layer_name.items():
        avg_embedding = encoder_hidden_states[layer_id].mean(dim=1).squeeze()
        featuremaps[layer_name].append(avg_embedding)
    for layer_id, layer_name in dec_id_to_layer_name.items():
        attention_pooling_embedding = decoder_hidden_states[0][layer_id]
        attention_pooling_embedding = attention_pooling_embedding.squeeze()
        featuremaps[layer_name].append(attention_pooling_embedding)
    return featuremaps


def eval_imdb(model, tokenizer, dataset, device):
    model.to(device)
    featuremaps = defaultdict(list)
    logger = get_logger(f"{__name__}.eval_imdb")
    for i, item in enumerate(dataset):
        inputs_ids = tokenizer.encode(item['text']+'</s>', return_tensors="pt").to(device)
        len_inputs_seq = len(inputs_ids[0])
        raw_outputs = model.generate(input_ids=inputs_ids, max_length=2, output_hidden_states=True, return_dict_in_generate=True)
        output_texts = [tokenizer.decode(ids) for ids in raw_outputs['sequences']]
        # logger.info(f"{i}th token len: {len_inputs_seq}, {i}th output text: {output_texts}")
        get_t5_featuremaps(featuremaps, raw_outputs['encoder_hidden_states'], raw_outputs['decoder_hidden_states'])
        
    enc_fm = {}
    for key, val in featuremaps.items():
        enc_fm[key] = torch.stack(val, dim=0)
    return enc_fm

def eval_race(model, tokenizer, dataset, device):
    model.to(device)
    featuremaps = defaultdict(list)
    logger = get_logger(f"{__name__}.eval_race")
    option_alphabet = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", \
        "(L)", "(M)", "(N)", "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]
    for i, item in enumerate(dataset):
        article = item['article']
        question = item['question']
        options = item['options']
        
        options_with_alphabet = " ".join([f"{alphabet} {option}" for alphabet, option in zip(option_alphabet, options)])
        context = f"{article} {options_with_alphabet}"
        input_text = f"question: {question} context: {context}"
        inputs_ids = tokenizer([input_text], return_tensors="pt").to(device)
        len_inputs_seq = len(inputs_ids[0])
        
        raw_outputs = model.generate(input_ids=inputs_ids['input_ids'], attention_mask=inputs_ids['attention_mask'], max_length=128, 
                                    output_hidden_states=True, return_dict_in_generate=True)
        output_texts = [tokenizer.decode(ids) for ids in raw_outputs['sequences']]
        logger.info(f"{i}th options: {options_with_alphabet}")
        logger.info(f"{i}th token len: {len_inputs_seq}, {i}th output text: {output_texts}")
        get_t5_featuremaps(featuremaps, raw_outputs['encoder_hidden_states'], raw_outputs['decoder_hidden_states'])
        
    enc_fm = {}
    for key, val in featuremaps.items():
        enc_fm[key] = torch.stack(val, dim=0)
    return enc_fm

def eval_qasc(model, tokenizer, dataset, device):
    """
    def get_response(question, context, max_length=64):
    input_text = 'question: %s  context: %s' % (question, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

    return tokenizer.decode(output[0])
    
    fact_1 = 'a watch is used for measuring time'
    fact_2 = 'Times are measured in seconds.'
    context = fact_1 + ' ' + fact_2
    question = 'What can be used to measure seconds? (A) Watch (B) seconds (C) fluid (D) Ruler (E) goggles (F) glasses (G) Drill (H) Scale'

    get_response(question, context)
    """
    model.to(device)
    featuremaps = defaultdict(list)
    logger = get_logger(f"{__name__}.eval_qasc")
    for i, item in enumerate(dataset):
        question = item['formatted_question']
        context =  " ".join([item["fact1"], item["fact2"], item["combinedfact"]])
        input_text = f"question: {question} context: {context}"
        inputs_ids = tokenizer([input_text], return_tensors="pt").to(device)
        len_inputs_seq = len(inputs_ids[0])
        
        raw_outputs = model.generate(input_ids=inputs_ids['input_ids'], attention_mask=inputs_ids['attention_mask'], max_length=128, 
                                    output_hidden_states=True, return_dict_in_generate=True)
        output_texts = [tokenizer.decode(ids) for ids in raw_outputs['sequences']]
        logger.info(f"{i}th options: {question}")
        logger.info(f"{i}th token len: {len_inputs_seq}, {i}th output text: {output_texts}")
        get_t5_featuremaps(featuremaps, raw_outputs['encoder_hidden_states'], raw_outputs['decoder_hidden_states'])
        
    enc_fm = {}
    for key, val in featuremaps.items():
        enc_fm[key] = torch.stack(val, dim=0)
    return enc_fm


def eval_multi_news(model, tokenizer, dataset, device):
    """
    def summarize(text, max_length=150):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds[0] 
    """
    model.to(device)
    featuremaps = defaultdict(list)
    logger = get_logger(f"{__name__}.eval_multi_news")
    for i, item in enumerate(dataset):
        document = item['document']
        if len(document) > 4096:
            document = document[:4096]
        inputs_ids = tokenizer.encode(document, return_tensors="pt", add_special_tokens=True).to(device)
        len_inputs_seq = len(inputs_ids[0])
        
        raw_outputs = model.generate(input_ids=inputs_ids, num_beams=2, max_length=128,  repetition_penalty=2.5, \
            length_penalty=1.0, early_stopping=True, output_hidden_states=True, return_dict_in_generate=True)
        output_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in raw_outputs['sequences']]
        logger.info(f"{i}th token len: {len_inputs_seq}, {i}th output text: {output_texts}")
        get_t5_featuremaps(featuremaps, raw_outputs['encoder_hidden_states'], raw_outputs['decoder_hidden_states'])
        
    enc_fm = {}
    for key, val in featuremaps.items():
        enc_fm[key] = torch.stack(val, dim=0)
    return enc_fm


def eval_squad(model, tokenizer, dataset, device):
    """
    def get_question(answer, context, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

    return tokenizer.decode(output[0])

    context = "Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
    answer = "Manuel"

    get_question(answer, context)
    """
    model.to(device)
    featuremaps = defaultdict(list)
    logger = get_logger(f"{__name__}.eval_squad")

    for i, item in enumerate(dataset):
        context = item['context']
        answer = item['answers']["text"][0]
        input_text = f"answer: {answer} context: {context} </s>"
        inputs_ids = tokenizer([input_text], return_tensors="pt").to(device)
        len_inputs_seq = len(inputs_ids[0])
        
        raw_outputs = model.generate(input_ids=inputs_ids['input_ids'], attention_mask=inputs_ids['attention_mask'], max_length=128, 
                                    output_hidden_states=True, return_dict_in_generate=True)
        output_texts = [tokenizer.decode(ids) for ids in raw_outputs['sequences']]
        logger.info(f"{i}th token len: {len_inputs_seq}, {i}th output text: {output_texts}")
        get_t5_featuremaps(featuremaps, raw_outputs['encoder_hidden_states'], raw_outputs['decoder_hidden_states'])
        
    enc_fm = {}
    for key, val in featuremaps.items():
        enc_fm[key] = torch.stack(val, dim=0)
    return enc_fm


def eval_common_gen(model, tokenizer, dataset, device):
    """
    def gen_sentence(words, max_length=32):
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

    return tokenizer.decode(output[0], skip_special_tokens=True)

    words = "tree plant ground hole dig"

    gen_sentence(words)
    """
    model.to(device)
    featuremaps = defaultdict(list)
    logger = get_logger(f"{__name__}.eval_common_gen")
    for i, item in enumerate(dataset):
        input_text = " ".join(item['concepts'])
        inputs_ids = tokenizer([input_text], return_tensors="pt").to(device)
        len_inputs_seq = len(inputs_ids[0])
        
        raw_outputs = model.generate(input_ids=inputs_ids['input_ids'], attention_mask=inputs_ids['attention_mask'], max_length=64, 
                                    output_hidden_states=True, return_dict_in_generate=True)
        output_texts = [tokenizer.decode(ids) for ids in raw_outputs['sequences']]
        logger.info(f"{i}th target sentence: {item['target']}")
        logger.info(f"{i}th token len: {len_inputs_seq}, {i}th output text: {output_texts}")
        get_t5_featuremaps(featuremaps, raw_outputs['encoder_hidden_states'], raw_outputs['decoder_hidden_states'])
        
    enc_fm = {}
    for key, val in featuremaps.items():
        enc_fm[key] = torch.stack(val, dim=0)
    return enc_fm