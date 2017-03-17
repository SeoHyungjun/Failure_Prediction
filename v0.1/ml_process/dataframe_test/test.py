#!/usr/bin/python3

import pandas as pd

rawData = pd.read_csv('example.csv')
print("Origin Data--------------------------")
print(rawData)
print("")

print("Duplicated--------------------------")
# duplicated=rawData.duplicated(subset='first_name', keep=False);
# duplicated(subset='column name', keep='first, last, False) 
# duplicated=rawData.duplicated(subset=['first_name', 'age'], keep='first');
# 인자가 없다면 모든 column이 중복되는 경우에 해당
duplicated=rawData.duplicated(keep=False)
print(duplicated)
print("")

#deduplicated=rawData.drop_duplicates)
print("drop_duplicates--------------------------")
#deduplicated=rawData.drop_duplicates(subset=['first_name', 'age'], keep='first')
deduplicated=rawData.drop_duplicates(subset=['first_name'], keep='first')
print(deduplicated)
print("")

print("drop_columns--------------------------")
#drop_columns = rawData.drop('last_name', axis=1)
drop_columns = rawData.drop(['last_name', 'age'], axis=1)
#drop_columns = rawData.drop('last_name', axis=1, inplace=True)
print(drop_columns)
print("")
#print(rawData)

print("drop_rows--------------------------")
#drop_rows = rawData.drop([0, 1], axis=0, inplace=False)
#drop_rows = rawData[rawData.first_name != 'Jake']
drop_rows = rawData[ (rawData.first_name != 'Jake') & (rawData.age != 24) ]
#drop_rows = rawData.drop(rawData[ (rawData.first_name == 'Jake') \
#                                    & (rawData.age == 24)].index) # 
#drop_rows = rawData.drop(rawData[ (rawData.first_name == 'Jake') \
#                                    | (rawData.age == 24)].index) # 
print(drop_rows)
print("")

print("reorder_columns--------------------------")
cols = rawData.columns.tolist()
# col_num = len(cols)
# if nth col move to end (n start at 0)
# cols = cols[:n] + cols[n+1:] + cols[n:n+1]
# cols = cols[:1] + cols[2:] + cols[1:2]
# cols = cols[:0] + cols[1:] + cols[0:1]
# print(rawData[cols[1:]])
# reorder_columns = rawData[cols]

Xdata = rawData[cols[:1] + cols[2:]]
Ydata = rawData[cols[1:2]]

print(Xdata)
print("")
print(Ydata)
print("")


print("reorder_columns--------------------------")
transposed = rawData.transpose()
print(transposed)

