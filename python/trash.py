class A:
    def __init__(self, a):
        self.a = a


a = A(1)

l = []
l.append(a)

a.a = 2

print(l[0].a)
