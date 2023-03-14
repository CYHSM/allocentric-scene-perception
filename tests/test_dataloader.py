import pytest
import helper

def test_all_valid_asp_datasets():
    all_valid = helper.all_valid_asp_datasets()
    assert len(all_valid) == 30