from pyenergyplus.model.model import EPBoolean


def to_epboolean(value: bool) -> EPBoolean:
    """Convert a standard Python boolean to an EPBoolean."""
    return EPBoolean.yes if value else EPBoolean.no
