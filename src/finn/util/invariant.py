def invariant(condition: bool, message: str):
    """This function acts as a convenient way to test for conditions and raise
    an error if the condition is not met. It also unifies manually raised errors
    and assert statements into one function call. It is inspired by its
    javascript counterpart (https://github.com/zertosh/invariant)"""

    if not condition:
        raise AssertionError(message)
