


# for multi feature simple models, we vectorize each and
# vectorizer = CountVectorizer()
# transformer = TfidfTransformer()
# title_vectors = vectorizer.fit_transform(df['title'][:10000])
# titile_tf_idf_vector = transformer.fit_transform(title_vectors)

# content_vectors = vectorizer.fit_transform(df['content'][:10000])
# content_tf_idf_vector = transformer.fit_transform(content_vectors)

# domain_vectors = vectorizer.fit_transform(df['domain'][:10000])
# domain_tf_idf_vector = transformer.fit_transform(domain_vectors)

# authors_vectors = vectorizer.fit_transform(df['authors'].apply(lambda x: ','.join(map(str, x))).str.lower().str.replace(" ", "-"))
# authors_tf_idf_vector = transformer.fit_transform(authors_vectors)

# x = np.concatenate([pd.DataFrame.sparse.from_spmatrix(titile_tf_idf_vector),
#                    pd.DataFrame.sparse.from_spmatrix(content_tf_idf_vector),
#                    pd.DataFrame.sparse.from_spmatrix(domain_tf_idf_vector)
#                    pd.DataFrame.sparse.from_spmatrix(domain_tf_idf_vector]), axis=1)





#overriding type
#df.loc[df['type'] == 'political', 'type'] = 'real'
#df.loc[df['type'] == 'clickbait', 'type'] = 'real'
#df.loc[df['type'] == 'reliable', 'type'] = 'real'




#Advanced model
"""
content_vectors = vectorizer.fit_transform(df['content'][:5000])
x1 = transformer.fit_transform(content_vectors)
y1 = df['target'][:5000]

x = torch.tensor(x1.toarray())
y = torch.tensor(y1.to_numpy())
print(x[:2])
print(y[:2])
"""



"""
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

#print(train_data[0][0].shape)
"""





"""
class MyDataset(Dataset):

  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)

    x=price_df.iloc[:,0:8].values
    y=price_df.iloc[:,8].values

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
"""