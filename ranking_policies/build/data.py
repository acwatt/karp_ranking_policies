# python 3.7
# file: data.py
"""Functions/classes to load and clean data."""
# Standard Library Imports

# Third-part Imports

# Local Imports / PATH changes
import pandas as pd

from ..config import PATHS

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
def load_andy() -> pd.DataFrame:
    """Return useable data cleaned by Andy for preliminary analysis."""
    data = pd.read_csv(PATHS.data.andyinput)
    data = data.drop(columns=['Unnamed: 0'])
    data['time2'] = data.time**2
    data['time2_lag'] = data.time_lag**2
    return data


def load_sunny() -> pd.DataFrame:
    """Return useable data for preliminary analysis."""
    data = pd.read_csv(PATHS.data.sunnyinput)
    # Subset of data between 1945 and 2005
    data = data.loc[(data.Year >= 1945) & (data.Year <= 2005)]
    data['time'] = data.time - 44
    data = data.sort_values(by=['time'])
    data = pd.get_dummies(data, columns=['group'])
    return data


# MAIN ========================================================================
if __name__ == '__main__':
    pass
