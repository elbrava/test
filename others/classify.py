from numpy.random import uniform

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s"
    , "t", "u", "v", "w", "x", "y", "z"]
inst = ""


def words():
    with open("words.txt", "w") as f:
        t = ""

        for l in letters:
            inst = ""
            t = inst
            t += l
            inst = t
            # print(t)
            for e in letters:

                t = inst
                t += e
                inst = l
                print(t)
                for r in letters:
                    t = inst
                    t += r
                    inst = l + e

                    f.write(t + "\n")


x_list = []
y_list = []
groups = letters


def mappings():
    with open("words.txt,""r") as f:
        words = f.read().split("/n")
        for _ in words:
            x_list.append(uniform(360))
            y_list.append(uniform(360))
