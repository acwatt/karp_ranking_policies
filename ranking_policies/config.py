# Python 3.7
# File name: 
# Authors: Aaron Watt
# Date: 2021-07-05
"""Module to be imported for project settings."""

# Standard library imports
from pathlib import Path
import sys


# CLASSES --------------------------
class Paths:
    """Inner paths class to store project paths commonly used.

    This will search the current working directory path for the name of the
    repo (beecensus). Since this code is only called from main.py, and main.py
    is inside the repo, it should be able to find the beecensus path.
    This also means the name of the repo cannot be changed.
    Since this is an inner class, paths will be accessible in the following way:
    Project = ProjectSettings()  # instance of the outer class
    Project.paths.root  # this will be the pathlib path to the github repo root
    """
    def __init__(self):
        # add root path of the project / git repo
        self.root = Path(*Path.cwd().parts[:Path.cwd().parts.index('karp_ranking_policies') + 1])
        # Top-level paths
        self.code = self.root / 'ranking_policies'
        self.docs = self.root / 'docs'
        self.output = self.root / 'output'
        # Data directories
        self.data = Data(self.root / 'data')


class Data:
    """Inner inner paths class to store data file paths."""
    def __init__(self, data_dir):
        self.root = data_dir
        self.sunny = self.root / 'sunny'
        self.tables = self.root / 'tables'
        self.temp = self.root / 'temp'
        # Lookup tables
        self.lookup_jpg = self.tables / 'tbl_jpg_lookup.csv'
        self.lookup_fips = self.tables / 'tbl_fips_lookup.csv'
        # Data files
        self.input = self.sunny / 'clean' / 'grouped_nation.1751_2014.csv'

# FUNCTIONS --------------------------


# MAIN -------------------------------
# Create instances of each class to be called from other
PATHS = Paths()


# OTHER GLOBALS -------------------------------


# REFERENCES -------------------------
"""

"""
