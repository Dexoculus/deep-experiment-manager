import os
import sys

# -- Path setup --------------------------------------------------------------
# Add project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)


# -- Project information -----------------------------------------------------
project = 'PyTorch Experiment Manager'
copyright = '2024, Dexoculus'
author = 'Dexoculus'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Automatically document modules/classes/functions
    'sphinx.ext.napoleon',      # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.autosummary',   # Generate summary tables for modules/classes
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and directories to ignore.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc options ---------------------------------------------------------
autodoc_member_order = 'bysource'  # Document members in the order they appear in the source code
autodoc_typehints = 'description'  # Show type hints in the description

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Autosummary options -----------------------------------------------------
autosummary_generate = True  # Automatically generate stub pages for autosummary

# -- Source Code Linking -----------------------------------------------------
# Enable linking to the source code in your documentation
html_show_sourcelink = True
