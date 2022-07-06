"""describe properties of amino acids"""

from __future__ import annotations


# Degine allowed protein sequence characters
# All canonical amino acids
ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

# Amino acid code conversion
AA_DICT = {"Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
           "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
           "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
           "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
           "Ter": "*"}

# the upper case three letter code for the amino acids 
ALL_AAS_TLC_DICT = {k.upper(): v for k, v in AA_DICT.items() if v != "*"}

# the upper case three letter code for the amino acids 
ALL_AAS_TLC = list(ALL_AAS_TLC_DICT.keys())

AA_NUMB = len(ALL_AAS)
ALLOWED_AAS = set(ALL_AAS)

# propertyies of the amino acids
AA_PROP = {
            "R": "Positive", 
            "H": "Positive", 
            "K": "Positive",
            "D": "Negative",
            "E": "Negative",
            "S": "Polar uncharged",
            "T": "Polar uncharged",
            "N": "Polar uncharged",
            "Q": "Polar uncharged",
            "C": "Special",
            # "U": "Special",
            "G": "Special",
            "P": "Special",
            "A": "Hydrophobic",
            "V": "Hydrophobic",
            "I": "Hydrophobic",
            "L": "Hydrophobic",
            "M": "Hydrophobic",
            "F": "Hydrophobic",
            "Y": "Hydrophobic",
            "W": "Hydrophobic"
        }