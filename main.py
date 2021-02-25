#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vocabularies import Vocab
from config import *
from dataset import Code2VecDataset, ShuffleDataset
from torch.utils.data import DataLoader
import torch as tt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import Code2Vec
from catalyst.runners import SupervisedRunner
from catalyst.dl import EarlyStoppingCallback
from catalyst.utils import plot_metrics


# In[2]:


dataset = "/home/sasha/MainProject/code2vec/data/Test/Test"


# In[3]:


with open(dataset + ".dict.c2v", "rb") as file:
    token_voc = Vocab(file, MAX_TOKEN_VOCAB_SIZE)
    paths_voc = Vocab(file, MAX_PATH_VOCAB_SIZE)
    target_voc = Vocab(file, MAX_TARGET_VOCAB_SIZE)


# In[4]:


test_dataset = ShuffleDataset(
    Code2VecDataset(dataset + ".test.c2v", token_voc, paths_voc, target_voc),
    SHUFFLE_BUFFER_SIZE
)

train_dataset = ShuffleDataset(
    Code2VecDataset(dataset + ".train.c2v", token_voc, paths_voc, target_voc),
    SHUFFLE_BUFFER_SIZE
)

valid_dataset = ShuffleDataset(
    Code2VecDataset(dataset + ".val.c2v", token_voc, paths_voc, target_voc),
    SHUFFLE_BUFFER_SIZE
)


# In[5]:


test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)


# In[6]:


model = Code2Vec(DEFAULT_EMBEDDINGS_SIZE,
                 MAX_TOKEN_VOCAB_SIZE,
                 MAX_PATH_VOCAB_SIZE,
                 MAX_TARGET_VOCAB_SIZE)

criterion = CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=LR_START)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2)  # TODO: keep?


# In[7]:


runner = SupervisedRunner(device=tt.device('cpu'))
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders={"train": train_loader, "valid": valid_loader},
    scheduler=scheduler,
    logdir='./logdir',
    num_epochs=NUM_TRAIN_EPOCHS,
    verbose=True,
    callbacks=[
        EarlyStoppingCallback(patience=PATIENCE, min_delta=PATIENCE_DELTA),
    ]
)


# In[ ]:


plot_metrics(
    logdir='./logdir',
    metrics=["loss"]
)


# In[ ]: