#!/bin/python

import json

nodes = list()
links = list()
node_num = 1000
rack_num = 1


for i in range(0, node_num) :
	if ( (i != 0) and (i % 10) == 0 ) :
		rack_num = rack_num + 1

	name = 'Server-' + str(rack_num) + '-'  + str((i%10) + 1)
	node = {'name' : name, 'group' : rack_num, 'status' : '0'}
	nodes.append(node)


node_num_in_group = 10
group_num = 100

# linking in group
for i in range(0, group_num) :

	if( i != group_num-1) :
		source = i*node_num_in_group
		target = source + node_num_in_group
		link = {'source' : source, 'target' : target}
		links.append(link)

	for j in range(0, node_num_in_group) :
		source = i * node_num_in_group
		target = source + j
		link = {'source' : source, 'target' : target}
		links.append(link)

data = {'nodes' : nodes, 'links' : links}

jsonstr = json.dumps(data, sort_keys=True, indent=4)
print jsonstr
