import csv
f = open('Disk_SMART_dataset.csv', 'r', encoding='utf-8')
lines = csv.reader(f)
for line in lines:
    print(line[0])
    if line[1] == " -1":
        filename = 'failure/F'
    elif line[1] == " +1":
        filename = 'normal/N'
                               
    filename = filename + '_' + str(line[0]) + '.csv'
    with open(filename, 'a') as sn_f:
        for i in range(0,14):
            if i != 13:
                sn_f.write(str(line[i]) + ',')
            else:
                sn_f.write(str(line[i]) + '\n')
f.close()
