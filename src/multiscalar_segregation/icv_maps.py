"""Category -> component-score maps for the ICV, per census year.

2010 maps score each category of the raw 2010 questionnaire variables.

2022 maps align the 2022 questionnaire categories to the identical score
semantics. New 2022 categories (absent in 2010) and their harmonization:

- V05=3  panel prefabricado (walls, 0.56% of occupied dwellings) -> 2/3
         (durable industrial material; scored with adobe/wood tier, below
         masonry, above unclad cane)
- V07=3  marmol/marmeton (floors, 0.59%) -> 1.0 (top tier)
- V11=3  biodigestor (sanitation, 0.20%) -> 1.0 on the 0-2 scale
         (functionally equivalent to a septic tank)
- V14=2  contenedor municipal (waste, 10.9%) -> 1.0 (municipal collection,
         equivalent to carro recolector)

Health coverage: the 2010 questionnaire asked about private insurance (P36) and
social-security affiliation (P35); the 2022 census asks only P30 (affiliation).
For 2010->2022 comparability the harmonized component uses affiliation alone in
BOTH years (P30 in {1..6} = covered; jubilados count as covered).
"""

MAPS_2010_VIVIENDA = {
    "roof":        ("V01", {"1": 1.0, "2": 0.5, "3": 0.5, "4": 1.0, "5": 0.0, "6": 0.0}),
    "walls":       ("V03", {"1": 1.0, "2": 1.0, "3": 2 / 3, "4": 2 / 3, "5": 1 / 3,
                            "6": 0.0, "7": 0.0}),
    "floor":       ("V05", {"1": 1.0, "2": 0.0, "3": 1.0, "4": 0.5, "5": 0.0,
                            "6": 0.0, "7": 0.0}),
    "water":       ("V08", {"1": 1.0, "2": 2 / 3, "3": 1 / 3, "4": 0.0}),
    "drainage":    ("V09", {"1": 2.0, "2": 1.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0}),
    "electricity": ("V10", {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 0.0}),
    "waste":       ("V13", {"1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0}),
}
MAPS_2010_HOGAR = {
    "kitchen":  ("H02", {"1": 1.0, "2": 0.0}),
    "bathroom": ("H03", {"1": 1.0, "2": 0.0, "3": 0.0}),
    "stove":    ("H05", {"1": 1.0, "2": 1.0, "3": 1.0, "4": 0.0, "5": 0.0,
                         "6": 0.0, "7": 0.0}),
    "phone":    ("H07", {"1": 1.0, "2": 0.0}),
    "cell":     ("H08", {"1": 1.0, "2": 0.0}),
    "internet": ("H09", {"1": 1.0, "2": 0.0}),
    "tv":       ("H11", {"1": 2.0, "2": 1.0}),
}

MAPS_2022_VIVIENDA = {
    "roof":        ("V03", {"1": 1.0, "2": 0.5, "3": 0.5, "4": 1.0, "5": 0.0, "6": 0.0}),
    # INEC added "panel prefabricado" in 2022, shifting madera to 5 and cana
    # revestida to 6 (categories 4 and 5 in 2010)
    "walls":       ("V05", {"1": 1.0, "2": 1.0, "3": 2 / 3, "4": 2 / 3, "5": 2 / 3,
                            "6": 1 / 3, "7": 0.0, "8": 0.0}),
    "floor":       ("V07", {"1": 1.0, "2": 1.0, "3": 1.0, "4": 0.5, "5": 0.0,
                            "6": 0.0, "7": 0.0, "8": 0.0}),
    "water":       ("V09", {"1": 1.0, "2": 2 / 3, "3": 1 / 3, "4": 0.0}),
    "drainage":    ("V11", {"1": 2.0, "2": 1.0, "3": 1.0, "4": 0.0, "5": 0.0,
                            "6": 0.0, "7": 0.0}),
    "waste":       ("V14", {"1": 1.0, "2": 1.0, "3": 0.0, "4": 0.0, "5": 0.0,
                            "6": 0.0, "7": 0.0}),
    # no electricity entry: the 2022 census splits it into V12 (public grid)
    # and V13 (other source), combined in 04_units_census.build_2022
}
MAPS_2022_HOGAR = {
    "kitchen":  ("H02", {"1": 1.0, "2": 0.0}),
    "bathroom": ("H03", {"1": 1.0, "2": 0.0, "3": 0.0}),
    "stove":    ("H05", {"1": 1.0, "2": 1.0, "3": 1.0, "4": 0.0, "5": 0.0,
                         "6": 0.0, "7": 0.0}),
    "phone":    ("H1001", {"1": 1.0, "2": 0.0}),
    "cell":     ("H1002", {"1": 1.0, "2": 0.0}),
    "tv":       ("H1003", {"1": 2.0, "2": 1.0}),  # INEC label: television pagada
    "internet": ("H1004", {"1": 1.0, "2": 0.0}),
}


P30_COVERED = {"1", "2", "3", "4", "5", "6"}
