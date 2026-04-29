# ~/gitrepos/palette/setup.py
from setuptools import setup, find_packages

setup(
    name="fisheye",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "palette-backfill-subject-body-mask-qc=fisheye.refinement.subject_body_mask_qc:main",
            "palette-write-refined-subject-mask-edit=fisheye.utils.write_refined_subject_mask_edit:main",
        ],
    },
    python_requires=">=3.11",
)
