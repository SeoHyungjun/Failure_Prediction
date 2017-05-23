#!/usr/bin/python3

def test_func(arg1, arg2, arg3=1):
    print(arg1, arg2, arg3)


args = [1, 2]
test_func(*args)