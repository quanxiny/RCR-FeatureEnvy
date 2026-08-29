from __future__ import annotations

import random


def project_name(label_line: str) -> str:
    return label_line.split()[0].split('/')[1].split('_')[0]


def ordered_projects(label_lines) -> list[str]:
    projects = []
    seen = set()
    for line in label_lines:
        project = project_name(line)
        if project not in seen:
            seen.add(project)
            projects.append(project)
    return projects


def published_project_folds(label_lines, split_seed=100, fold_count=5):
    """Reproduce the published project's exact project-level fold algorithm."""
    projects = ordered_projects(label_lines)
    random.Random(split_seed).shuffle(projects)
    fold_size = len(projects) // fold_count
    folds = {}
    for fold in range(1, fold_count + 1):
        start = (fold - 1) * fold_size
        stop = fold * fold_size
        test_projects = set(projects[start:stop])
        train_projects = set(projects) - test_projects
        folds[fold] = {"train": train_projects, "test": test_projects}
    return folds


def split_labels(label_lines, fold, split_seed=100):
    projects = published_project_folds(label_lines, split_seed=split_seed)[fold]
    train, test = [], []
    for line in label_lines:
        project = project_name(line)
        if project in projects["train"]:
            train.append(line)
        elif project in projects["test"]:
            test.append(line)
    return train, test


def inner_project_validation(label_lines, task, fold, fraction=0.10, seed=101):
    """Create a fixed validation subset from outer-training projects only."""
    projects = ordered_projects(label_lines)
    random.Random(seed + task * 10 + fold).shuffle(projects)
    validation_count = max(1, round(len(projects) * fraction))
    validation_projects = set(projects[:validation_count])
    train = [line for line in label_lines if project_name(line) not in validation_projects]
    validation = [line for line in label_lines if project_name(line) in validation_projects]
    return train, validation, validation_projects
