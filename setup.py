import sys
from setuptools import setup

setup(
      name = "sotabox-kdh",        # what you want to call the archive/egg
      version = "0.0.1",
      packages=["sotabox"],    # top-level python modules you can import like
      dependency_links = [],      # custom links to a specific project
      install_requires=[],
      extras_require={},      # optional features that other packages can require
      package_data = {},
      author="Daeha kim",
      author_email = "kdhht5022@gmail.com",
      description = "The first example program in Python",
      license = "MIT",
      keywords= "example documentation tutorial",
      url = "https://github.com/kdhht2334/sotabox",
      entry_points = {
      "console_scripts": [        # command-line executables to expose
                          "helloworld_in_python = helloworld.main:main",
                          ],
      "gui_scripts": []       # GUI executables (creates pyw on Windows)
      }
      )
