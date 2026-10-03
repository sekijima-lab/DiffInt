"""Preserve BioPython 1.79's standard amino-acid mapping and KeyError."""
from Bio.Data.PDBData import protein_letters_3to1


def three_to_one(residue):
    return protein_letters_3to1[residue]
