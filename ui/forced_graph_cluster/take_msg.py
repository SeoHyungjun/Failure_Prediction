#!/bin/python

import time
import re
import json

def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

if __name__ == '__main__':
    logfile = open("/var/log/messages","r")

    for line in follow(logfile):
        print line,
        status = line.split(".")[1].split(":")[0]
        if (status == "emerg" or status == "alert" or status == "crit" or status == "err" or status == "warning") :
            with open("test.json", "r+") as f :
                data = json.load(f)
                data['nodes'][0]['status'] = 1
                f.seek(0)
                f.write(json.dumps(data))
                f.truncate()

        '''p = re.compile('\D+[.]\D+:')
        m = p.match(line)
        if m :
            m.group()[0]
            print('Match found : ', m.group())
        else :
            print('No match')
        '''
        #print re.match("*[.]", line).group[0]
