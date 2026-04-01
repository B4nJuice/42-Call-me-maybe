import math

def fn_add_numbers(a: float, b: float) -> float:
    raise Exception("Test exception")
    return a+b


def fn_greet(name: str) -> str:
    greet: str = f"Hello {name}"
    print(greet)
    return greet


def fn_reverse_string(s: str):
    return s[::-1]


def fn_get_square_root(a: float) -> float:
    return math.sqrt(a)
