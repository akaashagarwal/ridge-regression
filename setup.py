"""Ridge regression."""
import textwrap

from setuptools import setup

setup(entry_points={"console_scripts": ["ridge-reg = src.__main__:main"]},
      use_scm_version={
          "write_to":
          "src/version.py",
          "write_to_template":
          textwrap.dedent('''
                  """Contains version information."""
                  from __future__ import unicode_literals
                  __version__ = "{version}"
                  ''').lstrip()})
