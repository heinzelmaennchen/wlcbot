
import pytest
from utils.deathroll_processing import format_num
import math

def test_format_num():
    """Test format_num formatting logic."""
    # Integers
    assert format_num(100) == "100"
    assert format_num(0) == "0"
    
    # Floats that are effectively integers (close enough)
    assert format_num(100.0) == "100"
    assert format_num(100.000000001) == "100"
    
    # Floats that should be formatted
    # default decimals=0 ? 
    # The function signature is def format_num(num, decimals=0):
    # If decimals=0, f"{num:.{decimals}f}" -> "10.5" formatted as "10" using .0f? rounded?
    # f"{10.5:.0f}" is "10" or "11"? Python 3 rounds to nearest even number for .5 usually?
    # Let's check implementation behavior expectation.
    # format_num(10.5) -> "10" or "11"
    
    # Passing decimals
    assert format_num(10.123, decimals=2) == "10.12"
    assert format_num(10.129, decimals=2) == "10.13"
    
    # None or NaN/Inf
    assert format_num(None) == "N/A"
    assert format_num(float('nan')) == "N/A"
    assert format_num(float('inf')) == "N/A"
    
    # String pass through?
    assert format_num("123") == "123"
