import lib

func_name = "print_lib"
args = [4, 2]

func = getattr(lib.library, func_name)

ret = func(*args)

print(ret)

func_name = "print_lib2"
func = getattr(lib.library, func_name)

ret = func(*args)
