from tkinter import HIDDEN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import psycopg2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib
import matplotlib.pyplot as plt

# Connect with database
conn = psycopg2.connect(
    "host=localhost dbname=moviedb user=laust1 password=123")
cur = conn.cursor()

# A query of all articles with all entities joined
try:
    cur.execute("""
    select article.id, title, type, domain, article_url, scraped_at, inserted_at, updated_at, a.name, keyword, tag, content  
	    from article
        inner join has_type ht on article.id = ht.article_id
        inner join type t on ht.type_id = t.id
	    left join written_by wb on article.id = wb.article_id
        left join author a on wb.author_id = a.id
	    left join has_tag hta on article.id = hta.article_id
        left join tag ta on hta.tag_id = ta.id
	    left join has_meta_keyword hmk on article.id = hmk.article_id
        left join meta_keyword mk on hmk.meta_keyword_id = mk.id
        where t.type = 'reliable' or t.type = 'political' or t.type = 'fake' or t.type = 'clickbait'
""")
except:
    # rollback in the case of stuck transaction
    print("rolled back")
    conn.rollback()

df = pd.DataFrame(cur.fetchall(), columns=['id', 'title', 'type', 'domain',
                                           'article_url', 'scraped_at', 'inserted_at',
                                           'updated_at', 'authors', 'keywords', 'tags',
                                           'content'])

# if blank values are NaN first replace to ''
df = df.fillna('')

# we group by id, since from our query, each article with e.g. tags will appear for each
# tag seperately, we implode on all columns using groupby function on all the attributes that
# will be the same for each duplicate and we use aggregate function for the features/columns
# that causes the duplication of tuples which include keywords, authors, and tags. Notice that keywords,
# authors, and tags are being duplicated themselves due to cross-product, thus we create a function
# that only includes unique values and does not contain Nan value.
df = df.groupby(['id', 'title', 'type', 'domain', 'article_url', 'scraped_at', 'inserted_at',
                 'updated_at', 'content']).agg({'keywords': lambda x: [x for x in list(set(x.tolist())) if str(x) != ''],
                                                'tags': lambda x: [x for x in list(set(x.tolist())) if str(x) != ''],
                                                'authors': lambda x: [x for x in list(set(x.tolist())) if str(x) != '']}
                                               ).reset_index()


df['target'] = np.where(df['type'] == 'fake', 1, 0)  # fake = 1, real = 0


# data splitting into training and test - only 1 feature which is 'content'
x = df['content'][:10000]  # content only


vectorizer = CountVectorizer()
transformer = TfidfTransformer()


x_df = df['content'][:500]  # , 'target']


x_vectorized = vectorizer.fit_transform(x_df)
x = transformer.fit_transform(x_vectorized).toarray()
y = df['target'][:500].to_numpy()


# split data into training, validation, and test data (features and labels, x and y)
split_idx = int(0.6*len(x))  # about 80%
train_x, remaining_x = x[:split_idx], x[split_idx:]
train_y, remaining_y = y[:split_idx], y[split_idx:]

test_idx = (len(x)-split_idx)//2
val_x, test_x = remaining_x[:test_idx-2], remaining_x[test_idx-2:-4]
val_y, test_y = remaining_y[:test_idx-2], remaining_y[test_idx-2:-4]

"""
# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
"""

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(
    train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 5

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, num_workers=0)
valid_loader = DataLoader(valid_data, shuffle=False,
                          batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_data, shuffle=False,
                         batch_size=batch_size, num_workers=0)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size())  # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)


class NeuralNet(nn.Module):
    '''Definition of Neural network architecture'''

    def __init__(self, input_size, hid_size):
        # Call base class constructor (this line must be present)
        super(NeuralNet, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_size, hid_size)    # 2 inputs to 1 output
        ###
        # ADD CODE HERE
        ###
        self.hid1 = nn.Linear(hid_size, hid_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hid_size, 1)    # 1 input to 1 output

    def forward(self, x):
        '''Define forward operation (backward is automatically deduced)'''
        x = self.layer1(x)
        ###
        # ADD CODE HERE
        ###
        #x = self.hid1(x)
        x = self.relu(x)
        x = self.layer2(x)
        #x = F.softmax(x, dim = 1)

        return x


# Instantiate model
inputsize = x.shape[1]
hidden_size = 8
model = NeuralNet(input_size=inputsize, hid_size=hidden_size)

# Define loss function
#loss_function = nn.CrossEntropyLoss()
loss_function = nn.BCEWithLogitsLoss()
# Instantiate optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(model, train_loader):
    '''One epoch of training'''

    # Switch to train mode
    model.train()

    loss_total = 0
    for batch_idx, (x, y) in enumerate(train_loader):

        # Make prediction from x (forward)
        y_pred = model(x.float())
        y_pred = torch.reshape(y_pred, (-1,))
        #print('y_pred:', y_pred)

        # Calculate loss
        loss = loss_function(y_pred.float(), y.float())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        loss.backward()

        # Let the optimizer update the parameters
        optimizer.step()

        loss_total += loss

    # Calculate average loss over entire dataset
    return loss_total/len(train_loader.dataset)


def test(model, valid_loader):
    '''One epoch of testing'''

    # Switch to test mode
    model.eval()

    loss_total = 0
    for batch_idx, (x, y) in enumerate(test_loader):

        # Make prediction from x (forward)
        y_pred = model(x.float())
        y_pred = torch.reshape(y_pred, (-1,))

        # Calculate loss
        loss = loss_function(y_pred.float(), y.float())

        loss_total += loss

    # Calculate average loss over entire dataset
    return loss_total/len(valid_loader.dataset)


epochs = 1000
for epoch in range(epochs):

    train_loss = train(model, train_loader)
    validation_loss = test(model, valid_loader)
    num_correct = 0

    if epoch % 10 == 0:
        print('Epoch {:4d}\t train loss: {:f}\t validation loss: {:f}'.format(
            epoch, train_loss.item(), validation_loss.item()))
