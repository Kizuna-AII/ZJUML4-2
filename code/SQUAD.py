import torch
import torch.utils
import random
import numpy as np
import transformers
import tqdm
import os
from transformers.data.processors.squad import SquadV2Processor

def evaluate(model, tokenizer, device, maxSequenceLength, maxQueryLength, documentStride):
    processor = SquadV2Processor()
    devData = processor.get_dev_examples(".")
    features, devDataset= transformers.squad_convert_examples_to_features(
        examples = devData,
        tokenizer = tokenizer,
        max_seq_length = maxSequenceLength,
        max_query_length = maxQueryLength,
        doc_stride = documentStride,
        return_dataset = "pt",
        threads = 1,
        is_training = False)
    batchSize = 2
    sampler = torch.utils.data.SequentialSampler(devDataset)
    dataLoader = torch.utils.data.DataLoader(devDataset, sampler = sampler, batch_size = batchSize)
    
    results = []
    for batch in tqdm.tqdm(devDataset):
        model.eval()
        batch = tuple(bat.to(device) for bat in batch)
        with torch.no_grad():
            startPosition = batch[3]
            outputs = model(input_ids = batch[0], attention_mask = batch[1], token_type_ids = batch[2])
        for i, index in enumerate(example):
            feature = features[index.item()]
            id = int(feature.unique_id)
            output = [output[i].detach().cpu().tolist() for output in outputs]
            print(len(output))
            results.append(transformers.data.processors.squad.SquadResult(id,output[0],output[2]))
    predictions = transformers.data.metrics.squad_metrics.compute_predictions_logits(
            devData, features, results)
    results = transformers.data.metrics.squad_metrics.squad_evaluate(devData, predictions)
    return results

if __name__ == "__main__" :
    documentStride = 128
    maxSequenceLength = 384
    maxQueryLength = 64
    outputPath = "./results/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpunum = torch.cuda.device_count()
    print("gpunum :",gpunum)
    # set the random seed of training
    seed = 142857
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_type = "bert"
    shortcut = "bert-base-uncased"
    config = transformers.AutoConfig.from_pretrained(shortcut)
    tokenizer = transformers.AutoTokenizer.from_pretrained(shortcut)
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(shortcut,config = config)
    model.to(device)
    # read in data
    processor = SquadV2Processor()
    trainData = processor.get_train_examples(".")
    features, trainDataset = transformers.squad_convert_examples_to_features(
        examples = trainData,
        tokenizer = tokenizer,
        max_seq_length = maxSequenceLength,
        max_query_length = maxQueryLength,
        doc_stride = documentStride,
        return_dataset = "pt",
        is_training = True)
    # build up model
    batchSize = 12
    trainSampler = torch.utils.data.RandomSampler(trainDataset)
    trainDataloader = torch.utils.data.DataLoader(trainDataset, sampler = trainSampler, batch_size = batchSize)
    trainEpoch = 10
    learningRate = 3e-5
    totalTrainingStep = len(trainDataloader) // trainEpoch
    optimizer = transformers.AdamW(params=[i for next, i in model.named_parameters()] ,lr = learningRate)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = totalTrainingStep)
    globalStep = 1
    epoch = 0
    trainingLoss = 0
    model.zero_grad()
    trainIterator = tqdm.trange(epoch, trainEpoch, desc = "Epoch", disable = False)
    for i in trainIterator:
        epochIterator = tqdm.tqdm(trainDataloader, desc = "Iteration", disable = False)
        for step, batch in enumerate(epochIterator):
            model.train()
            batch = tuple(bat.to(device) for bat in batch)
            inputs ={"input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            trainingLoss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            globalStep += 1
            results = evaluate(model, tokenizer, device, maxSequenceLength, maxQueryLength, documentStride)
    averageTrainingLoss = trainingLoss / globalStep