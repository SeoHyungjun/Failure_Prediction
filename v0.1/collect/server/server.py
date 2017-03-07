import socket, select

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 3140))
sock.listen(5)
sock.setblocking(0)

ep = select.epoll()
ep.register(sock.fileno(), select.EPOLLIN)

BUFSIZE = 1024
sock_map = {}
while True:
    events = ep.poll(1)
    for fd, evt in events:
        if fd == sock.fileno():
            csock, addr = sock.accept()
            csock.setblocking(0)
            ep.register(csock.fileno(), select.EPOLLIN)
            sock_map[csock.fileno()] = csock
            print("peer has opened connection, fileno: {}".format(csock.fileno()))
        elif evt & select.EPOLLHUP:
            ep.unregister(fd)
            sock_map[fd].close()
            del sock_map[fd]
            print("perr has closed connection, fileno: {}".format(fd))
        elif evt & select.EPOLLIN:
            buf = sock_map[fd].recv(BUFSIZE)
            print("recv : {}".format(buf))
            ep.modify(fd, select.EPOLLOUT)
        elif evt & select.EPOLLOUT:
            buf = b"hello epoll server!"
            written = sock_map[fd].send(buf)
            print("send : {}".format(written))

ep.unregister(sock.fileno())
ep.close()
sock.close()
