import socket

MSGLEN = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 3140))
msg = 'hello?'
sock.send(msg.encode('ascii'))
data = sock.recv(1024)
sock.close()
print(data.decode('ascii'))
