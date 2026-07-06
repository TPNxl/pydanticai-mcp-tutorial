# PDG 2022 values: mass in GeV/c², width in GeV
_PDG_DATA: dict[str, dict] = {
    "electron": {"mass_gev": 0.000510999, "width_gev": 0.0,       "charge": -1,   "spin": "1/2"},
    "muon":     {"mass_gev": 0.10566,     "width_gev": 2.996e-19, "charge": -1,   "spin": "1/2"},
    "z":        {"mass_gev": 91.1876,     "width_gev": 2.4952,    "charge":  0,   "spin": "1"},
    "w":        {"mass_gev": 80.377,      "width_gev": 2.085,     "charge": "±1", "spin": "1"},
    "higgs":    {"mass_gev": 125.20,      "width_gev": 0.00408,   "charge":  0,   "spin": "0"},
    "top":      {"mass_gev": 172.69,      "width_gev": 1.42,      "charge": "2/3","spin": "1/2"},
}

_ALIASES: dict[str, str] = {
    "h": "higgs", "h0": "higgs",
    "z0": "z", "z boson": "z",
    "w boson": "w", "w+": "w", "w-": "w",
    "t": "top", "t quark": "top", "top quark": "top",
    "e": "electron", "e-": "electron",
    "mu": "muon", "mu-": "muon",
}

def pdg_lookup(particle_name: str) -> dict:
    """Look up PDG properties (mass, width, charge, spin) for a particle by name.

    Args:
        particle_name: Common name or symbol of the particle
                       (e.g. 'electron', 'Z', 'Higgs', 'top', 'W', 'muon').
    """
    key = particle_name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _PDG_DATA:
        available = ", ".join(_PDG_DATA)
        return {"error": f"Unknown particle '{particle_name}'. Available: {available}"}
    return {"particle": key, **_PDG_DATA[key]}
