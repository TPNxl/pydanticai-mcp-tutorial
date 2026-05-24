# Goal: An MCP server exposing physics tools that any MCP client can call.
# It exposes the pdg_lookup function from pdg.py, and a simple list_files tool to list files in a directory.
# Note: the file lookup is not known by example 4 and as such the agent must query it.
# Also notice that if I add more tools, the agents can use them without any client-side changes.
# Run with: fastmcp run 5_mcp_server.py
# Or:       python 5_mcp_server.py

import os

from fastmcp import FastMCP

from pdg import pdg_lookup as _pdg_lookup

mcp = FastMCP("physics-tools")

@mcp.tool()
def pdg_lookup(particle_name: str) -> dict:
    """Look up PDG properties (mass, width, charge, spin) for a particle by name.

    Args:
        particle_name: Common name or symbol of the particle
                       (e.g. 'electron', 'Z', 'Higgs', 'top', 'W', 'muon').
    """
    return _pdg_lookup(particle_name)

@mcp.tool()
def list_files_in_curr_dir() -> list[str]:
    """List the names of files in the current directory.
    """
    try:
        return sorted(
            entry.name for entry in os.scandir(".") if entry.is_file()
        )
    except FileNotFoundError:
        return [f"Error: current directory not found...?"]

# TODO: note about guardrails
@mcp.tool()
def read_file_in_curr_dir(filename: str) -> str:
    """
    Read the contents of a file in the current directory.
    Args:
        filename: Name of the file to read (must be in the current directory).
    Returns: 
        The contents of the file, or an error message if the file is not found.
    """

    # 1. Check if the file is in the current directory (and NOT in a sub or superdirectory for safety)
    # Resolve to an absolute path and confirm it sits directly inside cwd (no traversal via "../")
    resolved = os.path.realpath(filename)
    cwd = os.path.realpath(".")
    if os.path.dirname(resolved) != cwd:
        return f"Error: '{filename}' is outside the current directory."
    if not os.path.isfile(resolved):
        return f"Error: file '{filename}' not found in current directory."

    # 2. Read and return the contents.
    with open(resolved) as f:
        return f.read()

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=23846)

