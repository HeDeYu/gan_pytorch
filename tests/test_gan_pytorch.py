#!/usr/bin/env python

"""Tests for `gan_pytorch` package."""

from gan_pytorch.gan_pytorch import sample


def test_sample():
    assert sample(True)
    assert not sample(False)
