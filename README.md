# To install:
```
conda create -n agentic-physics
conda activate agentic-physics
conda install python
pip install pydantic-ai
```
A reference Conda environment .yml file is included for cross-checking (``env1.yml``), but you are encouraged to install the latest version with the commands above. (Make sure you are in the ``agentic-physics`` Conda environment when you run it.)

To verify, run ``python 1_basics.py``.

# Goals for this tutorial

Get an API key, NOT a CLI from a real provider (solves ToS issues for work and integration issues)
Install a venv/conda env
Install dependencies
Then each of the below is a Python file (main, plus supporting files)
Call the LLM from the Python file using the library
Basic CoT: structured call-and-response with premade context
Agentic AI, implemented as functions and Python classes as a bare wrapper
Basic MCP server reading/writing files to a directory as plaintext
Basic MCP Python interpreter/terminal commands
And I think that's all we need