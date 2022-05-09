import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import *
import torch.nn as nn
import re
from string import digits
import string
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

NUMCODES = 0
vocab_size = 0

'''
This function is used to get the index of our text input sequence
'''
def to_index(sequence, token2idx):
    indices = []
    for word in sequence:
        idx = 0
        if word in token2idx:
            idx = token2idx[word]
        indices.append(idx)
    return indices

'''
This function is passed to our custom dataset, used to pad inputs and stack labels
'''
def collate_fn(data):
    text, labels = zip(*data)

    text = pad_sequence(text, batch_first=True)
    labels = torch.stack(labels)

    return text, labels

'''
This is our custom dataset class, used to represent a training our test dataset
'''
class CustomDataset(Dataset):
    
    def __init__(self, xdata, ydata):        
        # read in the data files
        self.datax = xdata
        self.datay = ydata 
        self.idx2word, self.word2idx = self.load_lookup()
        
    def load_lookup(self):
        """ load lookup for word """
        idx2token = {}
        i = 0
        for line in self.datax:
            line = line.strip()
            idx2token[i] = line
            i+=1
        token2idx = {w:i for i,w in idx2token.items()}
        return idx2token, token2idx    
    
    def __len__(self):
        
        # your code here
        return len(self.datax)
    
    def __getitem__(self, index):
        text = self.datax[index]
        print(text)
        text = to_index(text, self.word2idx)
        
        label = self.datay[index]
        label = label.type(torch.FloatTensor)
        # return text as long tensor, labels as float tensor;
        return torch.tensor(text, dtype=torch.long), label

'''
This is our model class that is described by the paper. 
'''
class ICD9PredCNN(nn.Module):
    def __init__(self,  kernel_size=(5,400), num_filter_maps=16, embed_size=400, dropout=0.5):
        super(ICD9PredCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(embed_size, 2 * embed_size, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(2 * embed_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(2 * embed_size, NUMCODES)


    def forward(self, x):
        x = self.embed(x)
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.softmax(self.fc1(x))
        return x

'''
The driving function where control flow occurs
''' 
def processData():
    #PREPROCESS BEGIN
    print("PREPROCESS BEGIN")
    data = pd.read_csv("NOTEEVENTS.csv", chunksize=100000)
    firstchunk = data.get_chunk(100000) #only grab the first chunk, we will use these patients as our dataset as doing it all is too big
    ids = firstchunk["SUBJECT_ID"].unique()
    patientToNotes= {}
    
    for i in range(0, ids.shape[0]):
        subject_id = ids[i]
        subject_rows = firstchunk.loc[firstchunk['SUBJECT_ID'] == subject_id]
        notes = subject_rows["TEXT"].values
        patientToNotes[subject_id] = notes

    patient_to_realnotes = {}
    iters = 0
    for key in patientToNotes.keys():
        for text in patientToNotes[key]:
            txt = re.search(r'Chief\sComplaint\:\n.*\n', text,re.IGNORECASE)
            if txt != None:
                txt = txt.group(0).lower()
                txt = re.sub(r'chief complaint:', "", txt)
                patient_to_realnotes[key] = txt
                break    
    
    data = pd.read_csv("DIAGNOSES_ICD.csv")
    patientToDiags = {}
    for i in range(0, ids.shape[0]):
        subject_id = ids[i]
        subject_rows = data.loc[data['SUBJECT_ID'] == subject_id]
        codes = subject_rows["ICD9_CODE"].unique()
        if subject_id in patient_to_realnotes:
            val = str(codes[0])
            if val.strip().lower() != 'nan':
                patientToDiags[subject_id] = codes[0]
            else:
                del patient_to_realnotes[subject_id]  
     
    allCodes = []
    for key in patientToDiags.keys():
        mainCode = patientToDiags[key]
        allCodes.append(mainCode)
    x = np.array(allCodes)
    uniqCodes= np.unique(x)
            
    target = torch.randint(0, len(uniqCodes), (len(uniqCodes),))
    one_hot = F.one_hot(target)

    NUMCODES = len(uniqCodes)
    remove_digits = str.maketrans('', '', digits)
    remove_punc = str.maketrans('','',string.punctuation)
    for key in patient_to_realnotes.keys():
        txt = patient_to_realnotes[key]
        txt = txt.translate(remove_digits)
        txt = txt.translate(remove_punc)
        txt = txt.strip()
        patient_to_realnotes[key] = txt   

    vocab_dict = {}
    for key in patient_to_realnotes.keys():
        txt = patient_to_realnotes[key]
        txt = txt.split(" ")
        for w in txt:
            if w not in vocab_dict:  
                vocab_dict[w] = 0
    vocab_size = len(vocab_dict)   

    #xvals and yvals are our final preprocessed data that will be used in the custom dataset
    xvals = list(patient_to_realnotes.values())
    yvals = []

    for code in allCodes:
        idx = uniqCodes.tolist().index(code)
        tens = one_hot[idx,]
        yvals.append(tens)
    #PREPROCESS END
    print("PREPROCESS END")
    train_size = int(len(yvals) * 0.7) #70-30 train-test split

    xvals_train = xvals[0:train_size]
    xvals_test = xvals[train_size-1:-1]
    yvals_train = yvals[0:train_size]
    yvals_test = yvals[train_size-1:-1]
    print("TRAIN BEGIN")
    runModelTrainAndEval(xvals_train, xvals_test, yvals_train, yvals_test)

'''
The function that creates the model and trains it
'''
def runModelTrainAndEval(xvals_train, xvals_test, yvals_train, yvals_test):
    train_set = CustomDataset(xvals_train, yvals_train)
    test_set = CustomDataset(xvals_test, yvals_test)
    train_loader = DataLoader(train_set, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)    

    model = ICD9PredCNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    n_epochs = 50
    train(model, train_loader, test_loader, n_epochs, optimizer, criterion)
    print("TRAIN END")

'''
This function evaluates the model based on the training, code has been reused partially from the CAML lab
'''
def eval(model, test_loader):

    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    model.eval()
    for sequences, labels in test_loader:
        y_hat = model(sequences)
        y_hat = (y_hat > 0.5).int()
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, labels.detach().to('cpu')), dim=0)
    
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return p, r, f

'''
This function trains the model, code has been reused partially from the CAML lab.  
'''
def train(model, train_loader, test_loader, n_epochs, optimizer, criterion):

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()

            y_hat, loss = None, None
            # your code here
            y_hat = model(sequences)
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        p, r, f = eval(model, test_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}'.format(epoch+1, p, r, f))

'''
Main function
'''
def main():
    processData()
    print("DONE")

main()