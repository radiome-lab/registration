import unittest

from radiome.core.utils.mocks import WorkflowDriver
from .utils import entry_dir, test_data_dir


class MyTestCase(unittest.TestCase):
    def test_run(self):
        wf = WorkflowDriver(entry_dir('ants'), test_data_dir())
        wf.linear = True
        wf.run(config={})


if __name__ == '__main__':
    unittest.main()
