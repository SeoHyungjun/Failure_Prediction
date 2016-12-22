import pandas as pd
import os

#files = [f for f in os.listdir('/home/syseng/ST4000DM000') if os.path.isfile(os.path.join('/home/syseng/ST4000DM000', f))]


data = pd.DataFrame()

#full_file_path = os.path.join('/home/syseng/ST4000DM000', file)
raw_data = pd.read_csv('/home/syseng/ST4000DM000/Z30252G1.csv', parse_dates=['date'], index_col='date')
tmp_data = pd.DataFrame()
tmp_data['serial_number'] = raw_data['serial_number']
tmp_data['5'] = raw_data['smart_5_normalized']
tmp_data['187'] = raw_data['smart_187_normalized']
tmp_data['188'] = raw_data['smart_188_normalized']
tmp_data['197'] = raw_data['smart_197_normalized']
tmp_data['198'] = raw_data['smart_198_normalized']

frames = [data, tmp_data]
data = pd.concat(frames)
#data.append(tmp_data, ignore_index=True)

#print(data.values)

deduped_data = data.drop_duplicates(keep='last')
#deduped_data.to_csv('deduped_smart_data.csv', sep=',')
print(deduped_data.values)


