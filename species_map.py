# species_map.py — Phase 2 complete
# Biological lookup: species → (class_label, evidence_type)
# IMPORTANT: folder names in your Data/ must match these keys exactly

SPECIES_MAP = {

    # ══ MICROBES — Fungi ════════════════════════════════════
    "Candida_albicans"           : ("fungi",            "microbe"),
    "Candida_glabrata"           : ("fungi",            "microbe"),

    # ══ MICROBES — Gram-negative bacilli ════════════════════
    "Escherichia_coli"           : ("gram_neg_bacilli", "microbe"),
    "Proteus_spp"                : ("gram_neg_bacilli", "microbe"),
    "Pseudomonas_aeruginosa"     : ("gram_neg_bacilli", "microbe"),

    # ══ MICROBES — Gram-positive bacilli ════════════════════
    "Clostridium_perfringens"    : ("gram_pos_bacilli", "microbe"),
    "Lactobacillus_casei"        : ("gram_pos_bacilli", "microbe"),
    "Listeria_monocytogenes"     : ("gram_pos_bacilli", "microbe"),

    # ══ MICROBES — Gram-negative cocci ══════════════════════
    "Neisseria_gonorrhoeae"      : ("gram_neg_cocci",   "microbe"),
    "Veillonella_spp"            : ("gram_neg_cocci",   "microbe"),

    # ══ MICROBES — Gram-positive cocci ══════════════════════
    "Staphylococcus_aureus"      : ("gram_pos_cocci",   "microbe"),
    "Staphylococcus_epidermidis" : ("gram_pos_cocci",   "microbe"),
    "Streptococcus_agalactiae"   : ("gram_pos_cocci",   "microbe"),
    # ⚠️ Note: Streptococcus NOT Staphylococcus agalactiae
    # Folder must be named Streptococcus_agalactiae

    # ══ DIATOMS — Pennate (bilaterally symmetric) ═══════════
    "Achnanthidium_minutissimum" : ("pennate",          "diatom"),
    "Cocconeis_placentula"       : ("pennate",          "diatom"),
    "Gomphonema_parvulum"        : ("pennate",          "diatom"),
    "Navicula_cryptocephala"     : ("pennate",          "diatom"),
    "Nitzschia_palea"            : ("pennate",          "diatom"),
    "Ulnaria_ulna"               : ("pennate",          "diatom"),

    # ══ DIATOMS — Centric (radially symmetric) ══════════════
    "Cyclotella_meneghiniana"    : ("centric",          "diatom"),
    "Melosira_varians"           : ("centric",          "diatom"),

    # ══ POLLENS — No class hierarchy at this scale ══════════
    "Callistemon_viminalis"      : ("pollen_species",   "pollen"),
    "Datura_innoxia"             : ("pollen_species",   "pollen"),
    "Citrus_aurantiifolia"       : ("pollen_species",   "pollen"),
    "Phlox_paniculata"           : ("pollen_species",   "pollen"),
    "Calliandra_haematocephala"  : ("pollen_species",   "pollen"),
    "Antirrhinum_majus"          : ("pollen_species",   "pollen"),
    "Clarkia_amoena"             : ("pollen_species",   "pollen"),
}

def get_class(species_name):
    """Returns (class_label, evidence_type) or ('unknown','unknown')."""
    return SPECIES_MAP.get(species_name, ("unknown", "unknown"))

def get_all_species_by_type(evidence_type):
    """Returns list of all species for a given evidence type."""
    return [sp for sp, (cls, et) in SPECIES_MAP.items() if et == evidence_type]