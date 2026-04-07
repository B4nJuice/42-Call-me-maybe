import math


def fn_add_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers.

    Parameters
    ----------
    a : float
        First value.
    b : float
        Second value.

    Returns
    -------
    float
        Sum of ``a`` and ``b``.
    """
    return a+b


def fn_greet(name: str) -> str:
    """Generate and print a greeting for a name.

    Parameters
    ----------
    name : str
        Name to greet.

    Returns
    -------
    str
        Greeting message.
    """
    greet: str = f"Hello {name}"
    print(greet)
    return greet


def fn_reverse_string(s: str) -> str:
    """Return the reversed version of a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Reversed string.
    """
    return s[::-1]


def fn_get_square_root(a: float) -> float:
    """Compute the square root of a number.

    Parameters
    ----------
    a : float
        Input value.

    Returns
    -------
    float
        Square root of ``a``.
    """
    return math.sqrt(a)
