import socket, select

class Collector():

    def __init__(self, bufsize):
        self.BUFSIZE = bufsize
        self.sock_map = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', 3140))
        self.sock.listen(5)
        self.sock.setblocking(0)
        self.ep = select.epoll()
        self.ep.register(self.sock.fileno(), select.EPOLLIN)

    def loop(self, recv_f, send_f):
        while True:
            events = self.ep.poll(1)
            for fd, evt in events:
                if fd == self.sock.fileno():
                    csock, addr = self.sock.accept()
                    csock.setblocking(0)
                    self.ep.register(csock.fileno(), select.EPOLLIN)
                    self.sock_map[csock.fileno()] = csock
                    test = csock.getsockname()
                    print("peer has opened connection, {}:{}".format(test, csock.fileno()))
                elif evt & select.EPOLLHUP:
                    self.ep.unregister(fd)
                    self.sock_map[fd].close()
                    del self.sock_map[fd]
                    print("perr has closed connection, fileno: {}".format(fd))
                elif evt & select.EPOLLIN:
                    buf = self.sock_map[fd].recv(self.BUFSIZE)
                    if len(buf) == 0:
                            self.ep.modify(fd, 0)
                            self.sock_map[fd].shutdown(socket.SHUT_RDWR)
                            continue
                    recv_f(self.sock_map[fd], buf)

                    buf = send_f(self.sock_map[fd])
                    if len(buf) == 0:
                            self.ep.modify(fd, 0)
                            self.sock_map[fd].shutdown(socket.SHUT_RDWR)
                            continue
                    written = self.sock_map[fd].send(buf)
                    #self.ep.modify(fd, select.EPOLLOUT)

                elif evt & select.EPOLLOUT:
                    print("select EPOLLOUT")

    def exit(self):
        self.ep.unregister(self.sock.fileno())
        self.ep.close()
        self.sock.close()

def recv_proc(sock, buf):
    print("recv : {}".format(buf))

def send_proc(sock):
    return b"hello epoll server!"

clt = Collector(1024)
clt.loop(recv_proc, send_proc)
clt.exit()
