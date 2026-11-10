"""The licence of the model weights, said on every run.

The weights were trained on ToothFairy data (CC BY-NC-SA 4.0). The dataset's
authors allow them to be distributed only as a separate asset under that
licence, with the licence stated and ToothFairy3 acknowledged whenever the
weights are used. So the engine says it in the log of every run and writes it
into report.json — whoever runs it from the command line, without the add-on,
sees it too. Do not remove.
"""

LICENSE = "CC BY-NC-SA 4.0"
SOURCE = "https://ditto.ing.unimore.it"

LINES = (
    f"model weights: {LICENSE} (https://creativecommons.org/licenses/by-nc-sa/4.0/), "
    "non-commercial use only",
    f"trained on ToothFairy2 / ToothFairy3 ({SOURCE}) and DentVoxel; "
    "cite the ToothFairy papers listed there",
)

REPORT = {
    "license": LICENSE,
    "commercial_use": False,
    "data_sources": ["ToothFairy3", "ToothFairy2", "DentVoxel"],
    "cite": SOURCE,
}
