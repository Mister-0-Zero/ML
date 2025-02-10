def accurancy(predict, real):
    accurancy_ = 0
    for p, r in zip(predict, real):
        p = 1 if p > 0.5 else 0
        if p == r:
            accurancy_ += 1
    accurancy_res = accurancy_ / len(predict)
    return accurancy_res