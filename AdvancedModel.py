import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import psycopg2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 

import matplotlib
import matplotlib.pyplot as plt

# Connect with database
conn = psycopg2.connect(
    "host=localhost dbname=sample_db user=anderssteiness password=XXX")
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
x = df['content'][:10000] #content only

# creating train and test set for mutiple feature simple models
df['Multiple Features'] = df['title'] + df['content'] + df['domain'] + df['authors'].apply(lambda x: ','.join(map(str, x))).str.lower().str.replace(" ", "-")
#x = df['Multiple Features'] #multiple meta-data



vectorizer = CountVectorizer()
transformer = TfidfTransformer()



xy_df = df[['content', 'target']][:5000]
xy_vectorized = vectorizer.fit_transform(xy_df)
xy = transformer.fit_transform(xy_vectorized).toarray()

train_split_fraction = 0.5
batch_size = 10

train_data = torch.utils.data.TensorDataset(
    *torch.split(torch.from_numpy(
        xy[:int(train_split_fraction*len(xy))]).float(), 2, dim=1))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True)
test_data = torch.utils.data.TensorDataset(
    *torch.split(torch.from_numpy(
        xy[int(train_split_fraction*len(xy)):]).float(), 2, dim=1))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data),
                                          shuffle=False)

print(train_data[0][0].shape)


class NeuralNet(nn.Module):
    '''Definition of Neural network architecture'''

    def __init__(self):
        # Call base class constructor (this line must be present)
        super(NeuralNet, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(2,1)    # 2 inputs to 1 output
        ###
        # ADD CODE HERE
        ###
        self.layer2 = nn.Linear(1,1)    # 1 input to 1 output

    def forward(self, x):
        '''Define forward operation (backward is automatically deduced)'''
        x = self.layer1(x) 
        ###
        # ADD CODE HERE
        ###
        x = self.layer2(x)

        return x

# Instantiate model
model = NeuralNet()

# Define loss function
loss_function = nn.MSELoss(reduction='sum')

# Instantiate optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def train(model, train_loader):
    '''One epoch of training'''

    # Switch to train mode
    model.train()

    loss_total = 0
    for batch_idx, (x,y) in enumerate(train_loader):

        # Make prediction from x (forward)
        y_pred = model(x)

        # Calculate loss
        loss = loss_function(y_pred, y)

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        loss.backward()

        # Let the optimizer update the parameters
        optimizer.step()

        loss_total += loss

    # Calculate average loss over entire dataset
    return loss_total/len(train_loader.dataset)


def test(model, test_loader):
    '''One epoch of testing'''

    # Switch to test mode
    model.eval()
    
    loss_total = 0
    for batch_idx, (x,y) in enumerate(test_loader):

        # Make prediction from x (forward)
        y_pred = model(x)
        
        # Calculate loss
        loss = loss_function(y_pred, y)
        
        loss_total += loss

    # Calculate average loss over entire dataset
    return loss_total/len(test_loader.dataset)




epochs = 1000
for epoch in range(epochs):

    train_loss = train(model, train_loader)
    test_loss = test(model, test_loader)

    if epoch %10 == 0:
        print('Epoch {:4d}\t train loss: {:f}\ttest loss: {:f}'.format(
            epoch, train_loss.item(), test_loss.item()))


