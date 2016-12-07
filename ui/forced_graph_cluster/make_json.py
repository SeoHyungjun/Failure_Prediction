#!/bin/python

import json
import random

nodes = list()
links = list()
groups = list()
rack_num = 1
sum_nodes = 0

group_num = 15


# create node(new) and linking node(except cluster linking) 
for rack_num in range(0, group_num) :
	node_num_in_group = random.randrange(6, 20) # determine node_num in this group
	groups.append({'group_num' : rack_num, 'start_node' : sum_nodes, 'node_num' : node_num_in_group,'graphed' : 0})

	for node_num in range (0, node_num_in_group) : # create node and linking in group
		name = 'Server-' + str(rack_num+1) + '-' + str(node_num+1)
		node = {'name' : name, 'group' : rack_num+1, 'status' : '0'}
		nodes.append(node)

		source = sum_nodes
		target = sum_nodes + node_num

		if source != target :
			link = {'source' : source, 'target' : target}
			links.append(link)

	sum_nodes = sum_nodes + node_num_in_group

# cluster graph linking

sum_nodes = 0
neighbors = 0
groups_in_network = 0

for rack_num in range(0, group_num) :
#	if groups_in_network == group_num :
#		break

	if neighbors == 0 :
		neighbors = random.randrange(3,7)
		source = groups[rack_num]['start_node']
		if(rack_num != 0 and groups[rack_num]['graphed'] == 0) :
			link = {'source' : before_source, 'target' : source}
			links.append(link)

		before_source = source
		groups_in_network += 1

	while True : #find to connect clustered graph
		select_group = random.randrange(0, group_num)
		if(groups[select_group]['graphed'] == 0) :
			groups[select_group]['graphed'] = 1
			target = groups[select_group]['start_node']
			link = {'source' : source, 'target' : target}
			links.append(link)
			neighbors = neighbors - 1
#			groups_in_network += 1
			break


data = {'nodes' : nodes, 'links' : links}

jsonstr = json.dumps(data, sort_keys=True, indent=4)
print jsonstr
