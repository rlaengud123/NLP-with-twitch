import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

train_df = pd.read_csv('doc_v7_train.txt', sep=',')
test_df = pd.read_csv('doc_v7_test.txt', sep=',')

train_df.rename(columns = {'polarity': 'label'}, inplace = True)
test_df.rename(columns = {'polarity': 'label'}, inplace = True)

class NsmcDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label

nsmc_train_dataset = NsmcDataset(train_df)
train_loader = DataLoader(nsmc_train_dataset, batch_size=4, shuffle=True, num_workers=0)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 50
total_loss = 0
total_len = 0
total_correct = 0


model.train()
log = pd.DataFrame(columns = ['epoch', 'iteration', 'train_loss', 'train_acc'])
count = 0
for epoch in range(epochs):
    
    for text, label in train_loader:
        optimizer.zero_grad()
        
        # encoding and zero padding
        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list = [e[0:512] + [0] * (512-len(e)) for e in encoded_list]
        
#         if len(padded_list) != 512:
#             print(padded_list)
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, total_correct/total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0
            log.loc[count] = (epoch+1,itr,total_loss/p_itr,total_correct/total_len)
            count += 1      

        itr+=1

    torch.save(model.state_dict(), './train/bert_classification_epoch_{}.ckpt'.format(epoch+1))
log.to_csv('./train/BERT_log.csv', encoding='utf-8-sig')








