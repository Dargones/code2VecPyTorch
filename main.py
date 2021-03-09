#!/usr/bin/env python
# coding: utf-8

# # Training a code2vec model on a custom dataset

# In[1]:


import pickle
from vocabularies import Vocab
from config import *
from dataset import IterableBaseC2VDataset, ShuffleDataset, BaseC2VDataset
from torch.utils.data import DataLoader
import torch as tt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import BaseCode2Vec
from catalyst.runners import SupervisedRunner
from catalyst.dl import utils, EarlyStoppingCallback
from catalyst.utils import plot_metrics, load_checkpoint, unpack_checkpoint
from metrics import *


# ## Creating token\path\target vocabularies and initializing the datasets:

# In[2]:


Config.DATASET = "/home/sasha/Desktop/Research/code2vec/data/java14mMed/java14mMed"
Config.TRAIN_DATA = Config.DATASET + ".train.c2v"
Config.TEST_DATA = Config.DATASET + ".test.c2v"
Config.VAL_DATA = Config.DATASET + ".val.c2v"
Config.MAX_TOKEN_VOCAB_SIZE = 0.95  # 90% of all tokens are retained (most frequent tokens are kept)
Config.MAX_PATH_VOCAB_SIZE = 0.90
Config.MAX_TARGET_VOCAB_SIZE = 0.95


# In[3]:


Vocab.prepare_for_file(Config.TRAIN_DATA, override=False)
token_voc = Vocab.tokens(Config.TRAIN_DATA, Config.MAX_TOKEN_VOCAB_SIZE)
path_voc = Vocab.paths(Config.TRAIN_DATA, Config.MAX_PATH_VOCAB_SIZE)
target_voc = Vocab.targets(Config.TRAIN_DATA, Config.MAX_TARGET_VOCAB_SIZE)


# In[4]:


Config.PROPERTIES = 0
Config.BATCH_SIZE = 256


# In[5]:


# train_dataset = ShuffleDataset(
#     IterableBaseC2VDataset(Config.TRAIN_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES, skipOOV=True),
#     Config.SHUFFLE_BUFFER_SIZE
# )
# train_dataset = BaseC2VDataset(Config.TRAIN_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES, skipOOV=True)
# val_dataset = BaseC2VDataset(Config.VAL_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES)
# test_dataset = BaseC2VDataset(Config.TEST_DATA, token_voc, path_voc, target_voc, Config.PROPERTIES)


# In[6]:


# pickle.dump(train_dataset, open(Config.TRAIN_DATA + ".vectorized", "wb"))
# pickle.dump(test_dataset, open(Config.TEST_DATA + ".vectorized", "wb"))
# pickle.dump(val_dataset, open(Config.VAL_DATA + ".vectorized", "wb"))


# In[7]:


train_dataset = pickle.load(open(Config.TRAIN_DATA + ".vectorized", "rb"))
test_dataset = pickle.load(open(Config.TEST_DATA + ".vectorized", "rb"))
val_dataset = pickle.load(open(Config.VAL_DATA + ".vectorized", "rb"))


# In[8]:


test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
# train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)


# ## Creating the model

# In[9]:


Config.DEFAULT_EMBEDDINGS_SIZE = 32
Config.LR_START = 0.01


# In[10]:


model = BaseCode2Vec(Config.DEFAULT_EMBEDDINGS_SIZE,
                 len(token_voc),
                 len(path_voc),
                 len(target_voc), Config.PROPERTIES)
Config.PROPERTIES = 0
criterion = CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=Config.LR_START)
scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=8)  # TODO: criteria of plateau seem murky here.


# ## Training the model
# ## NOTE: Change tt.device('cuda') to tt.device('cpu') if CUDA is not configured!

# In[11]:


runner = SupervisedRunner(device=tt.device('cuda'))


# In[12]:


Config.NUM_TRAIN_EPOCHS = 200
Config.PATIENCE = 8
Config.PATIENCE_DELTA = 0.0001


# In[13]:


runner.train(
    model=model,
    main_metric = "f-score",
    minimize_metric = False,
    criterion=criterion,
    optimizer=optimizer,
    loaders={"train": train_loader, "valid": val_loader}, # loaders={"train": train_loader},
    scheduler=scheduler,
    logdir='./logdir',
    num_epochs=Config.NUM_TRAIN_EPOCHS,
    verbose=True,
    callbacks=[
        EarlyStoppingCallback(patience=Config.PATIENCE, min_delta=Config.PATIENCE_DELTA),
        SubtokenFScoreallback(target_vocab=target_voc)
    ]
)


# ## Training graphs and results on the testing set

# In[14]:


plot_metrics(
    logdir='./logdir',
    metrics=["f-score"],
    step="epoch"
)


# In[15]:


plot_metrics(
    logdir='./logdir',
    metrics=["loss"],
    step="epoch"
)


# In[16]:


checkpoint = load_checkpoint("logdir/checkpoints/last.pth")
unpack_checkpoint(checkpoint=checkpoint, model=model)


# In[17]:


predictions = runner.predict_loader(model=model, loader=test_loader)  # loader = test_loader
precision, recall, f_score = get_metrics_dataset(predictions, test_loader, target_voc)  # test_loader
print("precision = %.3f, recall = %.3f, f-score = %.3f" %(precision, recall, f_score))


# In[18]:


checkpoint = load_checkpoint("logdir/checkpoints/best.pth")
unpack_checkpoint(checkpoint=checkpoint, model=model)


# In[19]:


predictions = runner.predict_loader(model=model, loader=test_loader)  # loader = test_loader
precision, recall, f_score = get_metrics_dataset(predictions, test_loader, target_voc)  # test_loader
print("precision = %.3f, recall = %.3f, f-score = %.3f" %(precision, recall, f_score))


# In[21]:


from metrics import logger
import logging
logger.setLevel(logging.DEBUG)


# In[ ]:


predictions = runner.predict_loader(model=model, loader=test_loader)  # loader = test_loader
precision, recall, f_score = get_metrics_dataset(predictions, test_loader, target_voc)  # test_loader
print("precision = %.3f, recall = %.3f, f-score = %.3f" %(precision, recall, f_score))


# In[ ]:



