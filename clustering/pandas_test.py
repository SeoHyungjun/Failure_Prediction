import pandas as pd
import os

files = [f for f in os.listdir('/home/syseng/ST4000DM000') if os.path.isfile(os.path.join('/home/syseng/ST4000DM000', f))]


data = pd.DataFrame()

for file in files :
	try :
		print(file)
		full_file_path = os.path.join('/home/syseng/ST4000DM000', file)
		raw_data = pd.read_csv(full_file_path, parse_dates=['date'], index_col='date')
		tmp_data = pd.DataFrame()
		tmp_data['serial_number'] = raw_data['serial_number']
		tmp_data['5'] = raw_data['smart_5_normalized']
		tmp_data['187'] = raw_data['smart_187_raw']
		tmp_data['188'] = raw_data['smart_188_raw']
		tmp_data['197'] = raw_data['smart_197_raw']
		tmp_data['198'] = raw_data['smart_198_raw']
		frames = [data, tmp_data]
		data = pd.concat(frames)
		data = data.drop_duplicates(keep='last')
	except :
		print("error occured\n")

try :
	deduped_data = data.drop_duplicates(keep='last')
	deduped_data.to_csv('deduped_smart_data.csv', sep=',')
#print(deduped_data.values)
except :
	print("csv write error occur")

