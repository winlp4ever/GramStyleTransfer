import numpy as np

def sec2time(secs):
    secs = int(secs)
    if secs < 60:
        return "{}s".format(secs)
    if secs < 60 * 60:
        return "{}'{}s".format(secs // 60, secs % 60)
    return "{}h{}'{}s".format(secs // (60 * 60), secs % (60 * 60) // 60, secs % 60)

if __name__ == '__main__':
    print(sec2time(4600))