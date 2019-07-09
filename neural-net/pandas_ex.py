import pandas as pd
index1 = [1, 2, 3, 4]
columns1 = ['e', 'b', 'c']
data1 = [
    ['a1', 'b1', 'c1'],
    ['a2', 'b2', 'c2'],
    ['a3', 'b3', 'c3'],
    ['a4', 'b4', 'c4']]


index2 = [1, 4]
columns2 = ['b', 'c', 'd', 'e']
data2 = [
    ['b1', 'c1', '<D1', 'e1'],
    ['b4', '<C4', 'd4', 'e4']]

df1 = pd.DataFrame(index=index1, columns=columns1, data=data1)
df2 = pd.DataFrame(index=index2, columns=columns2, data=data2)

df1 = df1.reindex(columns = df2.columns.values, fill_value = 0)
print(df1)

# df1.update(df2)
# df1 = df1.merge(df2, how='left')
# print(df1)

# print(df2)