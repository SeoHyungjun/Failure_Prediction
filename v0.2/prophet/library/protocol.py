import struct

# Protocol module
#
#        +----------------+
#        | Protocol class |
#        +----------------+
#                |
# +------------------------------+
# | ProtocolVariableString class |  variable string support
# +------------------------------+
#                |
#  +-----------------------------+
#  | ProtocolVariableArray class |  variable array support
#  +-----------------------------+
#                |
#   +-------------------------+ byte, char, uchar, bool, short, ushort,
#   |  ProtocolBaseData class | long, ulong, float, double, string, array
#   +-------------------------+ support.
#

class BaseData():

    def __init__(self, type, value=0):
        self.value = value
        self.type = type
        self.size = struct.calcsize(self.type)
        pass

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

    def pack(self):
        return struct.pack('!' + self.type, self.value)

    def unpack(self, buffer):
        self.value = struct.unpack('!' + self.type, buffer[:self.size])[0]
        return self.size

class ArrayData():

    def __init__(self, type, len, value=[]):
        self.type = type
        self.len = len
        self.value = []
        self.size = struct.calcsize(self.type) * self.len
        self.set(value)

    def __setitem__(self, key, value):
         self.value.insert(key, BaseData(self.type, value))
 
    def __getitem__(self, key):
         return self.value[key].get()

    def set(self, value):
        self.value = []
        for i in range(self.len):
            self.value.append(BaseData(self.type, 0 if len(value)<=i else value[i]))

    def get(self):
        list = []
        for elem in self.value:
            list.append(elem.get())
        return list

    def pack(self):
        buffer = bytes()
        for elem in self.value:
            buffer += elem.pack()
        return buffer

    def unpack(self, buffer):
        for elem in self.value:
            cnt = elem.unpack(buffer)
            buffer = buffer[cnt:]
        return self.size

class VariableArrayData(ArrayData):

    def __init__(self, type, len=0, value=[]):
        super().__init__(type, len, value)
        pass

    def set(self, value):
        self.value = []
        self.value_len = BaseData('i', len(value))
        self.len = len(value)
        self.size = self.len * struct.calcsize(self.type)
        super().set(value)

    def pack(self):
        buffer = bytes()
        buffer += self.value_len.pack()
        buffer += super().pack()
        return buffer

    def unpack(self, buffer):
        cnt = self.value_len.unpack(buffer)
        self.set(list(range(self.value_len.get())))
        buffer = buffer[cnt:]
        super().unpack(buffer)
        return self.size

class VariableStringData():
 

    def __init__(self, value="", charset='utf-8'):
        self.set(value, charset)
 
    def set(self, value, charset='utf-8'):
        self.charset = charset
        print(type(str))
        if type(value) == str:
            value = value.encode(self.charset)
        self.len = BaseData('i', len(value))
        self.str = BaseData('%ss'%self.len.get(), value)
        self.size = self.len.size + self.str.size
 
    def get(self):
        self.str.value = self.str.value.decode(self.charset)
        return self.str.get()
 
    def pack(self):
        buffer = bytes()
        buffer += self.len.pack()
        buffer += self.str.pack()
 
        return buffer 
 
    def unpack(self, buffer):
        len = self.len.unpack(buffer)
        buffer = buffer[len:]
 
        self.str.type = '%ss'%self.len.get()
        len += self.str.unpack(buffer)
        return len 

class Protocol():

    def __init__(self):
        self.buffer = dict()
        self.keys = []
        self.types = []
        pass

    def append_field(self, field_list):
        for key in field_list:
            self.keys.append(key[0])
            self.types.append(key[1])
            if 's' == key[1]:
                self.buffer[key[0]] = ProtocolDataString()
            else:
                self.buffer[key[0]] = ProtocolData(key[1])

    def set(self, key, value):
        if key in self.buffer:
            index = self.keys.index(key)

            if 's' == self.types[index]:
                self.buffer[key] = ProtocolDataString(value)
            else:
                self.buffer[key] = ProtocolData(self.types[index], value)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        data = self.buffer.get(key)
        if data == None:
            return None

        return data.get()

    def print(self):
        for key in self.keys:
            print("{} : {}".format(key, self.buffer[key].get()))

    def tobytes(self):
        buffer = bytearray()
        for key in self.keys:
            buffer += self.buffer[key].pack()
        return buffer

    def frombytes(self, buffer):
        for key in self.keys:
            cnt = self.buffer[key].unpack(buffer)
            buffer = buffer[cnt:]
        pass


# ptl = Protocol()
# 
# ptl.append_field([("test", 'i')])
# ptl.append_field([("test2", 'f')])
# ptl.append_field([("string", 's')])
# ptl["test"] = 0x30
# ptl["test2"] = 1.2
# ptl["string"] = "hoochoona"
# 
# buffer = ptl.tobytes()
# 
# ptl2 = Protocol()
# 
# ptl2.append_field([("test", 'i')])
# ptl2.append_field([("test2", 'f')])
# ptl2.append_field([("string", 's')])
# 
# ptl2.frombytes(buffer)
# 
# ptl2.print()
