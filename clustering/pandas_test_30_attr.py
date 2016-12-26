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
		tmp_data['1'] = raw_data['smart_1_normalized']
		tmp_data['2'] = raw_data['smart_2_normalized']
		tmp_data['3'] = raw_data['smart_3_normalized']
		tmp_data['4'] = raw_data['smart_4_normalized']
		tmp_data['5'] = raw_data['smart_5_normalized']
		tmp_data['7'] = raw_data['smart_7_normalized']
		tmp_data['8'] = raw_data['smart_8_normalized']
		tmp_data['9'] = raw_data['smart_9_normalized']
		tmp_data['10'] = raw_data['smart_10_normalized']
		tmp_data['11'] = raw_data['smart_11_normalized']
		tmp_data['12'] = raw_data['smart_12_normalized']
		tmp_data['183'] = raw_data['smart_183_normalized']
		tmp_data['184'] = raw_data['smart_184_normalized']
		tmp_data['187'] = raw_data['smart_187_normalized']
		tmp_data['188'] = raw_data['smart_188_normalized']
		tmp_data['189'] = raw_data['smart_189_normalized']
		tmp_data['190'] = raw_data['smart_190_normalized']
		tmp_data['191'] = raw_data['smart_191_normalized']
		tmp_data['192'] = raw_data['smart_192_normalized']
		tmp_data['193'] = raw_data['smart_193_normalized']
		tmp_data['194'] = raw_data['smart_194_normalized']
		tmp_data['195'] = raw_data['smart_195_normalized']
		tmp_data['196'] = raw_data['smart_196_normalized']
		tmp_data['197'] = raw_data['smart_197_normalized']
		tmp_data['198'] = raw_data['smart_198_normalized']
		tmp_data['199'] = raw_data['smart_199_normalized']
		tmp_data['200'] = raw_data['smart_200_normalized']
		tmp_data['240'] = raw_data['smart_240_normalized']
		tmp_data['241'] = raw_data['smart_241_normalized']
		tmp_data['242'] = raw_data['smart_242_normalized']
		frames = [data, tmp_data]
		data = pd.concat(frames)
		data = data.drop_duplicates(keep='last')
		data.to_csv('30_attr_normalized.csv', sep=',')
	except :
		print("error occured\n")

#print(deduped_data.values)

