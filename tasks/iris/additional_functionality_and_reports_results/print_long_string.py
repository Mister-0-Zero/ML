
def print_long_string(Lstring):
    print()
    i = 0
    flag = 0
    while i < len(Lstring):
        if i % 100 == 0 and i != 0:
            flag = 1
        if flag and Lstring[i] == " ":
            flag = 0
            print()
            i += 1
            continue
        print(Lstring[i], end="")
        i +=1

    print("\n")