import csv
import os
from typing import Optional

import torchio as tio


def get_file_name(
    basename: str, extensions: list[str] = [".nii", ".nii.gz"]
) -> Optional[str]:
    for ext in extensions:
        filepath = f"{basename}{ext}"
        if os.path.exists(filepath):
            return filepath
    return None


def make_subject(
    folder_path: str,
    train_test_split: str = "train",
    extension: list[str] = [".nii", ".nii.gz"],
    add_heirarchical=False,
    add_binary=False,
) -> tio.Subject:
    # patient_id_str is formatted "MBAS_002"
    patient_id_str = os.path.basename(folder_path)
    patient_id = int(patient_id_str.split("_")[-1])
    mri = tio.ScalarImage(
        path=get_file_name(
            os.path.join(folder_path, f"{patient_id_str}_gt"), extension
        ),
        name="mri",
    )

    subject = tio.Subject(
        mri=mri,
        patient_id=patient_id,
        patient_id_str=patient_id_str,
        train_test_split=train_test_split,
    )
    label_path = get_file_name(
        os.path.join(folder_path, f"{patient_id_str}_label"), extension
    )
    if label_path is not None and os.path.exists(label_path):
        subject.add_image(
            tio.LabelMap(
                path=label_path,
                name="label",
            ),
            "label",
        )

    if add_heirarchical:
        label_path = get_file_name(
            os.path.join(folder_path, f"{patient_id_str}_hierarchical_label"),
            extension,
        )
        if label_path is not None and os.path.exists(label_path):
            subject.add_image(
                tio.LabelMap(
                    label_path,
                    name="hierarchical_label",
                ),
                "hierarchical_label",
            )
    if add_binary:
        label_path = get_file_name(
            os.path.join(folder_path, f"{patient_id_str}_binary_label"),
            extension,
        )
        if label_path is not None and os.path.exists(label_path):
            subject.add_image(
                tio.LabelMap(
                    path=label_path,
                    name="binary_label",
                ),
                "binary_label",
            )
    return subject


def load_label_csv(
    csv_path: str = "/home/bryan/data/brain_tumor/classification/train_labels.csv",
) -> dict[str, int]:
    with open(csv_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        rows = [row for row in csv_reader]
        # first row is just headers: BraTS21ID,MGMT_value
        label_map = {str(row[0]): int(row[1]) for row in rows[1:]}
    return label_map


def get_subject_folders(dataset_folder: str) -> list[str]:
    patient_folders = []
    for file_name in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file_name)
        if os.path.isdir(file_path):
            patient_files = os.listdir(file_path)
            # check if folder contains a MBAS_XXX_gt.nii.gz file
            for x in patient_files:
                if x.endswith(".nii.gz"):
                    patient_folders.append(file_path)
                    break
    return patient_folders


def load_subjects(
    dataset_folder: str,
    add_heirarchical=False,
    add_binary=False,
) -> list[tio.Subject]:

    train_folders = get_subject_folders(os.path.join(dataset_folder, "Training"))
    val_folders = get_subject_folders(os.path.join(dataset_folder, "Validation"))

    train_subjects = [
        make_subject(
            x, "train", add_heirarchical=add_heirarchical, add_binary=add_binary
        )
        for x in train_folders
    ]
    val_subjects = [
        make_subject(
            x, "validation", add_heirarchical=add_heirarchical, add_binary=add_binary
        )
        for x in val_folders
    ]
    subjects = train_subjects + val_subjects
    subjects = sorted(subjects, key=lambda x: x.patient_id)
    return subjects


def nifti_count_train_subjects(dataset_folder: str) -> int:
    subject_folders = get_subject_folders(os.path.join(dataset_folder, "Training"))
    return len(subject_folders)
