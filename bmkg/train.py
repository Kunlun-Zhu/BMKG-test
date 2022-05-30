import sys
from os.path import dirname

if __name__ == '__main__':
    sys.path.insert(0, dirname(dirname(__file__)))
    from bmkg import train_inner
    train_inner.main()
