import threading

answers = []
letters = "qwertyuiopasdfghjklzxcvbn"
numbers = "1234567890"
cha_racters = r"`~!@#$%^&*()_+[];''\,./{}:""|<>?"

_iter = 1
all_characters = letters + letters.upper() + numbers + cha_racters
all_list = []
with open("keys.text", "w") as fil:
    fil.write("\n".join(all_characters))


def main():
    global answers
    with open("keys.text", "r") as fi:
        answers = fi.read().split("\n")
    for an in answers:
        for key in all_characters:
            answers.append(an + key)
    with open(f"keys.text", "w") as f:
        f.write("\n".join(answers))
        print(answers)
        main()  # Press Shift+F10 to execute it or replace it with your code.


# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settin
if __name__ == '__main__':
    t = threading.Thread(target=main)
    t.daemon = True
    t.start()
