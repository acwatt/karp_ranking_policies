# python 3.7
# file: data.py
"""Functions/classes to load and clean data."""
# Standard Library Imports
import pandas as pd

# Third-part Imports

# Local Imports / PATH changes
import ranking_policies.build.data as bd

# Authorship
__author__ = "Aaron Watt, Larry Karp"
__copyright__ = "Copyright 2021, ACWatt"
__credits__ = ["Aaron Watt", "Larry Karp"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Aaron Watt"
__email__ = "aaron@acwatt.net"
__status__ = "Prototype"


# FUNCTIONS ===================================================================
def main():
    data_andy = bd.load_andy()
    print(data_andy.head())
    est.estimation_andy(data_andy)


# MAIN ========================================================================
if __name__ == '__main__':
    main()
