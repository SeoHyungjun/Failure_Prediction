class Protocol():

    def __init__(self):
        self.buffer = []
        pass

    def append(self, list):
        self.buffer[len(self.buffer):] = list

    def print(self):
        for elem in self.buffer:
            print("{}".format(elem))

    def encode(self):
        debuffer = []
        for elem in self.buffer:
            debuffer[len(debuffer):] = elem.encode('ascii')
        return debuffer

ptl = Protocol()

ptl.append(["ab", "ac", "ad", "ae"])
ptl.append(["ba", "bc", "bd", "be"])
ptl.print()

print("{}".format(ptl.encode()))
