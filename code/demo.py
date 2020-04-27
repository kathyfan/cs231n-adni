import random
import numpy as np

def demo_numpy():
    a = np.ones((3, 2))
    print(a)

def demo_random():
    """
        demo of random
    """
    # possible outcomes: 1, 2, 3, 4, 5, 6
    roll = random.randint(1, 6)
    print("roll: " + str(roll))

    # output between (0, 1)
    fraction = random.random()
    print("fraction: " + str(fraction))

    # output between (86, 100)
    grade = random.uniform(86, 100)
    print("grade: " + str(grade))

def demo_equals():
    """
        demo of ==
        """
    x = 5
    y = 10
    x_equals_y = (x == y)
    print("Are values of x and y the same? " + str(x_equals_y))
    print("Currently x = " + str(x) + ", y = " + str(y))
    x = y
    print("We just assigned x the same value as y. Now, x= " + str(x) + ", y = " + str(y))
    x_equals_y = (x == y)
    print("Are values of x and y the same? " + str(x_equals_y))

def main():
    demo_random()
    demo_equals()

if __name__ == '__main__':
    main()