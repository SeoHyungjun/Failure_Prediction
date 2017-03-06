##DEV_ENV Install sequence
sequence1. '# ./install_docker.sh' <br />
sequence2. '# ./make_dev_image.sh' <br />
####Caution!
1. While './install_docker.sh', one time must put the sudo passward when asked. <br />
2. User who execute install must have valid private-public key set in ~/.ssh directory. <br />
<br />
<br />

##Docker control([] mean short key)
exit container with container stop : '[ctrl] + [D]'  <br />
exit container without container stop : '[ctrl] + [P] + [ctrl] + [Q]' <br /> 

print all images created : '# docker images' <br />
print all container cmd : '# docker ps -a' <br />
restart container cmd : '# docker start -i NAMES' <br />
