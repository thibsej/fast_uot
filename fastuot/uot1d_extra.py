def sparsify_support(spt, down=True):
    spts = []
    k = 0
    while k < (len(spt) - 1):
        a, b = spt[k], spt[k + 1]
        if down and (a == (0,1)) and (b == (1,0)):
            spts.append((1,1))
            k = k+2
        if (not down) and (a == (1,0)) and (b == (0,1)):
            spts.append((1, 1))
            k = k + 2
        else:
            spts.append(a)
            k = k + 1
    if k == len(spt) - 1:
        spts.append(b)
    return spts