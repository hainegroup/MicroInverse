#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `MicroInverse` package."""


import unittest
import os
#from click.testing import CliRunner

#import MicroInverse.MicroInverse as mi
#
#mi.run_examples(['example1'])
#
#from MicroInverse import mutils #MicroInverse
from MicroInverse.MicroInverse_examples import run_examples
#from MicroInverse import cli
#
#run_examples(['example1'])
#
class TestMicroinverse(unittest.TestCase):
    """Tests for `MicroInverse` package."""
    
    def test_000(self):
        dirpath = os.path.dirname(os.path.abspath(__file__))+'/'
        assert run_examples([''], just_a_test=True) == 1
        assert run_examples(['example1'], just_a_test=False, datapath=dirpath) == 2

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
