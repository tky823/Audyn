import csv
from typing import List, Tuple

import torch

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "stratified_split",
]

primary_labels = [
    "asbfly",
    "ashdro1",
    "ashpri1",
    "ashwoo2",
    "asikoe2",
    "asiope1",
    "aspfly1",
    "aspswi1",
    "barfly1",
    "barswa",
    "bcnher",
    "bkcbul1",
    "bkrfla1",
    "bkskit1",
    "bkwsti",
    "bladro1",
    "blaeag1",
    "blakit1",
    "blhori1",
    "blnmon1",
    "blrwar1",
    "bncwoo3",
    "brakit1",
    "brasta1",
    "brcful1",
    "brfowl1",
    "brnhao1",
    "brnshr",
    "brodro1",
    "brwjac1",
    "brwowl1",
    "btbeat1",
    "bwfshr1",
    "categr",
    "chbeat1",
    "cohcuc1",
    "comfla1",
    "comgre",
    "comior1",
    "comkin1",
    "commoo3",
    "commyn",
    "compea",
    "comros",
    "comsan",
    "comtai1",
    "copbar1",
    "crbsun2",
    "cregos1",
    "crfbar1",
    "crseag1",
    "dafbab1",
    "darter2",
    "eaywag1",
    "emedov2",
    "eucdov",
    "eurbla2",
    "eurcoo",
    "forwag1",
    "gargan",
    "gloibi",
    "goflea1",
    "graher1",
    "grbeat1",
    "grecou1",
    "greegr",
    "grefla1",
    "grehor1",
    "grejun2",
    "grenig1",
    "grewar3",
    "grnsan",
    "grnwar1",
    "grtdro1",
    "gryfra",
    "grynig2",
    "grywag",
    "gybpri1",
    "gyhcaf1",
    "heswoo1",
    "hoopoe",
    "houcro1",
    "houspa",
    "inbrob1",
    "indpit1",
    "indrob1",
    "indrol2",
    "indtit1",
    "ingori1",
    "inpher1",
    "insbab1",
    "insowl1",
    "integr",
    "isbduc1",
    "jerbus2",
    "junbab2",
    "junmyn1",
    "junowl1",
    "kenplo1",
    "kerlau2",
    "labcro1",
    "laudov1",
    "lblwar1",
    "lesyel1",
    "lewduc1",
    "lirplo",
    "litegr",
    "litgre1",
    "litspi1",
    "litswi1",
    "lobsun2",
    "maghor2",
    "malpar1",
    "maltro1",
    "malwoo1",
    "marsan",
    "mawthr1",
    "moipig1",
    "nilfly2",
    "niwpig1",
    "nutman",
    "orihob2",
    "oripip1",
    "pabflo1",
    "paisto1",
    "piebus1",
    "piekin1",
    "placuc3",
    "plaflo1",
    "plapri1",
    "plhpar1",
    "pomgrp2",
    "purher1",
    "pursun3",
    "pursun4",
    "purswa3",
    "putbab1",
    "redspu1",
    "rerswa1",
    "revbul",
    "rewbul",
    "rewlap1",
    "rocpig",
    "rorpar",
    "rossta2",
    "rufbab3",
    "ruftre2",
    "rufwoo2",
    "rutfly6",
    "sbeowl1",
    "scamin3",
    "shikra1",
    "smamin1",
    "sohmyn1",
    "spepic1",
    "spodov",
    "spoowl1",
    "sqtbul1",
    "stbkin1",
    "sttwoo1",
    "thbwar1",
    "tibfly3",
    "tilwar1",
    "vefnut1",
    "vehpar1",
    "wbbfly1",
    "wemhar1",
    "whbbul2",
    "whbsho3",
    "whbtre1",
    "whbwag1",
    "whbwat1",
    "whbwoo2",
    "whcbar1",
    "whiter2",
    "whrmun",
    "whtkin2",
    "woosan",
    "wynlau1",
    "yebbab1",
    "yebbul3",
    "zitcis1",
]
num_primary_labels = len(primary_labels)


def stratified_split(path: str, train_ratio: float, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    g = torch.Generator()
    g.manual_seed(seed)

    filenames = {primary_label: [] for primary_label in primary_labels}
    train_filenames = []
    validation_filenames = []

    with open(path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx < 1:
                continue

            primary_label, *_, filename = line
            filenames[primary_label].append(filename)

    # split dataset
    for primary_label, _filenames in filenames.items():
        num_files = len(_filenames)
        indices = torch.randperm(num_files).tolist()

        for idx in indices[: int(train_ratio * num_files)]:
            train_filenames.append(_filenames[idx])

        for idx in indices[int(train_ratio * num_files) :]:
            validation_filenames.append(_filenames[idx])

    return train_filenames, validation_filenames
