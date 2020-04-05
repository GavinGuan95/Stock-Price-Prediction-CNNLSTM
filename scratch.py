# from collections import Iterator
class Count:
    """Iterator that counts upward forever."""

    def __init__(self, start=0):
        self.num = start

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        if num == 5:
            raise TypeError
        return num
a = Count(0)
b = a.__next__()
print(b)
print(a.__next__())