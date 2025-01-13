from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from transformers import get_linear_schedule_with_warmup

import os
import torch
import numpy as np
import random
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, RandomSampler, SequentialSampler
from nlp_dataset import generate_sample

# standard PyTorch approach of loading data in using a Dataset class.
class NAR_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.input_ids = []
        self.attn_masks = []

        for data_point in data:
            encodings = tokenizer.encode_plus(data_point,
                                              truncation=True,
                                              padding='max_length',
                                              max_length=max_length,
                                              # return a PyTorch tensor
                                              return_tensors='pt'       
                                            )
            self.input_ids.append(torch.squeeze(encodings['input_ids'],0))
            self.attn_masks.append(torch.squeeze(encodings['attention_mask'],0))


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.attn_masks[idx]


device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name: ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model_name = "gpt2-large" 
model_save_path = './model'

batch_size = 16
max_length = 102

configuration = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

model = model.to(device)

num_cats = 8
query_type = "sum"
num_query_cats = 4
train_low = 0
train_high = 5
test_low = 0
test_high = 20
num_train_samples = 50000
num_test_samples = 100
num_val_samples = 100
num_final_test_samples = 1000


def form_string(sample: tuple[list, float], isTrain: bool) ->tuple[str, float]:
    sample_text = sample[0]
    sample_ans = sample[1]
    input_list = [x if type(x) == str else format(x, '05.2f') for x in sample_text[:-1]]
    prompt = "<|startoftext|>" + ", ".join(input_list) + ". " + sample_text[-1] + "."
    if isTrain:
        prompt += " Answer: " + format(sample_ans, '06.2f')
        prompt += "<|endoftext|>"
    # else:
    #     prompt += " Answer: "
    return prompt, sample_ans

def get_metrics(predictions, target):
    diff_mse, off = [], 0
    diff_mape = []
    for (i, pred) in enumerate(predictions):
        try:
            diff_mse.append(abs(float(pred) - target[i])**2)
            diff_mape.append(abs(float(pred) - target[i]) / target[i])
        except:
            off += 1
    
    if len(diff_mse) == 0:
        return np.inf, np.inf, off / len(predictions) * 100
    
    return sum(diff_mse)/len(diff_mse), round(sum(diff_mape)/len(diff_mape) * 100, 2), round(off / len(predictions) * 100, 2)

for cur_iter in range(10):
    train_data_comb = [form_string(generate_sample(num_cats, query_type, train_low, train_high, num_query_cats), True) for _ in range(num_train_samples)]
    train_data = [x[0] for x in train_data_comb]
    train_data_ans = [x[1] for x in train_data_comb]
    test_data_comb = [form_string(generate_sample(num_cats, query_type, test_low, test_high, num_query_cats, train=False), False) for _ in range(num_test_samples)]
    test_data = [x[0] for x in test_data_comb]
    test_data_ans = [x[1] for x in test_data_comb]
    val_data_comb = [form_string(generate_sample(num_cats, query_type, train_low, train_high, num_query_cats, train=True), False) for _ in range(num_val_samples)]
    val_data = [x[0] for x in val_data_comb]
    val_data_ans = [x[1] for x in val_data_comb]
    test_data_all, test_data_all_ans = [], []
    for i in range(1, 11):
        cur = [form_string(generate_sample(num_cats, query_type, 0, 5 * i, num_query_cats, train=False), False) for _ in range(num_final_test_samples)]
        test_data_all.append([x[0] for x in cur])
        test_data_all_ans.append([x[1] for x in cur])

    print(val_data[0])
    print(val_data_ans[0])

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name,
                                                  bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>',
                                                  unk_token='<|unknown|>',
                                                  pad_token='<|pad|>'
                                                )

    dataset_indist_train = NAR_Dataset(train_data, tokenizer)
    dataset_indist_val = NAR_Dataset(val_data, tokenizer)
    dataset_ood = NAR_Dataset(test_data, tokenizer)
    
    print(f"input_ids: {dataset_ood[0][0]} attn_masks: {dataset_ood[0][1]}")
    print(tokenizer.decode(dataset_indist_train[0][0]))
    print(tokenizer.decode(dataset_indist_train[10][0]))

    train_dataloader = DataLoader(
                dataset_indist_train, 
                sampler = RandomSampler(dataset_indist_train),
                batch_size = batch_size # Trains with this batch size.
            )

    # Get valiation samples sequentially.
    validation_dataloader = DataLoader(
                dataset_indist_val, 
                sampler = SequentialSampler(dataset_indist_val),
                batch_size = batch_size # Evaluate with this batch size.
            )

    test_dataloader = DataLoader(
                dataset_ood, 
                sampler = SequentialSampler(dataset_ood),
                batch_size = batch_size # Evaluate with this batch size.
            )
                
    configuration = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    epochs = 3
    learning_rate = 2e-5
    warmup_steps = 1e2
    # to prevent any division by zero in the implementation
    epsilon = 1e-8
    optim = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

    total_steps = len(train_dataloader) * epochs  # [no batches] x [no epochs]

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    def infer(prompt):
        input = prompt
        input = tokenizer(input, return_tensors="pt")
        input_ids      = input["input_ids"]
        attention_mask = input["attention_mask"]

        output = model.generate(input_ids.to(device),
                                attention_mask=attention_mask.to(device),
                                max_new_tokens=15,
                                do_sample = True, top_k = 1, top_p = 0.85, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        start_index = output.find("Answer: ") + len("Answer: ")
        # Extract the first 5 characters from that point
        result = output[start_index:start_index + 6]

        return result

    def infer2(prompt):
        input = prompt
        input = tokenizer(input, return_tensors="pt")
        input_ids      = input["input_ids"]
        attention_mask = input["attention_mask"]

        output = model.generate(input_ids.to(device),
                                attention_mask=attention_mask.to(device),
                                max_new_tokens=15,
                                do_sample = True, top_k = 1, top_p = 0.85, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output[0], skip_special_tokens=True)

        return output


    for epoch_i in range(0, epochs):
        total_train_loss = 0
        model.train() 

        for step, batch in enumerate(train_dataloader): 
            b_input_ids = batch[0].to(device) 
            b_labels    = batch[0].to(device)
            b_masks     = batch[1].to(device) 

            model.zero_grad()
            outputs = model( input_ids = b_input_ids, labels = b_labels,
                            attention_mask = b_masks, token_type_ids = None )

            loss = outputs[0]

            # Get sample every x batches.
            if step % 100 == 0 and not step == 0:
                print("Evaluation:")
                model.eval()
                test_preds = [infer(test_data[i]) for i in range(len(test_data))]
                test_metrics = get_metrics(test_preds, test_data_ans)
                print(f"Test MSE Loss: {test_metrics[0]}, Test MAPE Loss: {test_metrics[1]}, Test Off: {test_metrics[2]}")
                val_preds = [infer(val_data[i]) for i in range(len(val_data))]
                val_metrics = get_metrics(val_preds, val_data_ans)
                print(f"Val MSE Loss: {val_metrics[0]}, Val MAPE Loss: {val_metrics[1]}, Val Off: {val_metrics[2]}")
                ind = random.randint(0, len(val_data) - 1)
                print(infer2(val_data[ind]))
                print("Actual: ", val_data_ans[ind])
                model.train()

            loss.backward()
            optim.step()
            scheduler.step()

    filename_mape = "./losses_mape_sum.txt"
    filename_mse = "./losses_mse_sum.txt"
    os.makedirs(os.path.dirname(filename_mape), exist_ok=True) 
    os.makedirs(os.path.dirname(filename_mse), exist_ok=True)

    model.eval()
    for i in range(10):
        test_preds = [infer(test_data_all[i][j]) for j in range(len(test_data_all[i]))]
        test_metrics = get_metrics(test_preds, test_data_all_ans[i])
        print(f"Test MSE Loss: {test_metrics[0]}, Test MAPE Loss: {test_metrics[1]}, Test Off: {test_metrics[2]}")
        with open(filename_mape, "a") as f:
            f.write(f"{test_metrics[1]}\t")
        with open(filename_mse, "a") as f:
            f.write(f"{test_metrics[0]}\t")

    with open(filename_mape, "a") as f:
        f.write("\n")
    with open(filename_mse, "a") as f:
        f.write("\n")
    
    print(f"Finished iteration {cur_iter+1}")
