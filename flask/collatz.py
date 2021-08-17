import matplotlib.pyplot as plt

def collatz(n):
    path = []
    while True:
        if n == 1 or n == -1 or n == -5 or n == -17:
            path.append(n)
            break
        elif n % 2 == 0:
            path.append(n)
            n = n//2
        else:
            path.append(n)
            n = 3 * n + 1
    return path

