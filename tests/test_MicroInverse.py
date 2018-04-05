#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `MicroInverse` package."""


import unittest
#from click.testing import CliRunner

#import MicroInverse.MicroInverse as mi
#
#mi.run_examples(['example1'])
#
#from MicroInverse import mutils #MicroInverse
from MicroInverse import run_examples
#from MicroInverse import cli
#
#run_examples(['example1'])
#
class TestMicroinverse(unittest.TestCase):
    """Tests for `MicroInverse` package."""
    
    def test_000(self):
        run_examples(['example1'])
        print('passed')

if __name__ == '__main__':
    unittest.main()

    #def setUp(self):
    #    """Set up test fixtures, if any."""
    #
    #def tearDown(self):
    #    """Tear down test fixtures, if any."""
    #
    #def test_000_something(self):
    #    """Test something."""
    #
    #def test_command_line_interface(self):
    #    """Test the CLI."""
    #    runner = CliRunner()
    #    result = runner.invoke(cli.main)
    #    assert result.exit_code == 0
    #    assert 'MicroInverse.cli.main' in result.output
    #    help_result = runner.invoke(cli.main, ['--help'])
    #    assert help_result.exit_code == 0
    #    assert '--help  Show this message and exit.' in help_result.output
