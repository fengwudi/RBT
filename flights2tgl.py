import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

df = pd.read_csv('Flights.csv')
edges = {
    'src': df['source'],
    'dst': df['destination'],
    'time': df['timestamp'],
    'edge_features': df['w']
}
node_features = df['state_label']
node_features = torch.tensor(node_features, dtype=torch.float32)
df_edges = pd.DataFrame(edges)
df_edges = df_edges.sort_values(['time', 'src'])
# ext_roll划分数据集,70%-15%-15%
ext_train_set, ext_val_set = train_test_split(df_edges, test_size=0.3, random_state=1, shuffle=False)
ext_val_set, ext_test_set = train_test_split(ext_val_set, test_size=0.5, random_state=1, shuffle=False)
# 转化成dataframe格式
ext_train_set = pd.DataFrame(ext_train_set)
ext_val_set = pd.DataFrame(ext_val_set)
ext_test_set = pd.DataFrame(ext_test_set)
# 编号，为了保存为edges.csv的格式
ext_train_set['ext_roll'] = 0
ext_val_set['ext_roll'] = 1
ext_test_set['ext_roll'] = 2
# int_roll划分数据集,60%-20%-20%
int_train_set, int_val_set = train_test_split(ext_train_set, test_size=0.4, random_state=1)
int_val_set, int_test_set = train_test_split(int_val_set, test_size=0.5, random_state=1)
# 转化成dataframe格式
int_train_set = pd.DataFrame(int_train_set)
int_val_set = pd.DataFrame(int_val_set)
int_test_set = pd.DataFrame(int_test_set)
# 编号，为了保存为edges.csv的格式
int_train_set['int_roll'] = 0
int_val_set['int_roll'] = 1
int_test_set['int_roll'] = 2
ext_val_set['int_roll'] = 3
ext_test_set['int_roll'] = 3

edges = pd.concat([int_train_set, int_val_set, int_test_set, ext_val_set, ext_test_set])
edges = edges.sort_values(['time', 'src'])
edges = edges.reset_index(drop=True)
edge_features = edges['edge_features'].values
edge_features = np.array(edge_features)

# feat_list = [float(0) for i in range(172)]
# print(feat_list)
# cnt = 0
for i in edge_features:
    edge_features[i] = torch.tensor(edge_features)
    # cnt += 1

# print(cnt)

edge_features = edge_features.reshape(-1, 1)
edge_features = torch.tensor(edge_features, dtype=torch.float32)
final_edges = edges.drop(['edge_features'], axis=1)

final_edges.to_csv('./DATA/edges.csv', index=True)
torch.save(edge_features, './DATA/edge_features.pt')
# torch.save(node_features, './2tgl/node_features.pt')


# print(final_edges)
# print(edge_features)