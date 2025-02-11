def accuracy(predict, real):
    accuracy_ = 0
    for p, r in zip(predict, real):
        p = 1 if p > 0.5 else 0
        if p == r:
            accuracy_ += 1
    accuracy_res = accuracy_ / len(predict)
    return accuracy_res