import os


def test_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def entry_dir(destination):
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'radiome', 'workflows', 'registration',
                     destination))
