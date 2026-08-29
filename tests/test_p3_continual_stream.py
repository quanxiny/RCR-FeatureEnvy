import unittest

from src.continual.stream import build_continual_stream


def synthetic_rows(projects=6, cases_per_project=3, samples_per_case=4):
    train = []
    index = 0
    for project_index in range(projects):
        project = f"project{project_index}"
        for case_index in range(cases_per_project):
            case_id = f"case_{project_index}_{case_index}"
            for sample_index in range(samples_per_case):
                label = str(sample_index % 2)
                train.append({
                    "task": "1", "fold": "1", "split": "train",
                    "split_project": project, "case_id": case_id,
                    "sample_id": f"sample_{index}", "label": label,
                    "state": "pre" if label == "1" else "post",
                })
                index += 1
    test = [{
        "task": "1", "fold": "1", "split": "test",
        "split_project": "heldout", "case_id": "case_heldout",
        "sample_id": "sample_heldout", "label": "1", "state": "pre",
    }]
    return train, test


class ContinualStreamTests(unittest.TestCase):
    def test_stream_is_deterministic_and_case_preserving(self):
        train, test = synthetic_rows()
        first = build_continual_stream(
            train, test, task=1, fold=1, seed=42,
            increment_count=3, probe_fraction=0.10,
        )
        second = build_continual_stream(
            train, test, task=1, fold=1, seed=42,
            increment_count=3, probe_fraction=0.10,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["audit"]["status"], "passed")
        locations = {}
        for increment in first["increments"]:
            for case_id in increment["train_case_ids"]:
                locations.setdefault(case_id, []).append((increment["id"], "train"))
            for case_id in increment["probe_case_ids"]:
                locations.setdefault(case_id, []).append((increment["id"], "probe"))
        self.assertTrue(all(len(location) == 1 for location in locations.values()))

    def test_projects_are_disjoint_and_samples_are_exhaustive(self):
        train, test = synthetic_rows(projects=9, cases_per_project=5)
        stream = build_continual_stream(
            train, test, task=1, fold=1, seed=43,
            increment_count=3, probe_fraction=0.20,
        )
        project_sets = [set(item["projects"]) for item in stream["increments"]]
        self.assertTrue(project_sets[0].isdisjoint(project_sets[1]))
        self.assertTrue(project_sets[0].isdisjoint(project_sets[2]))
        self.assertTrue(project_sets[1].isdisjoint(project_sets[2]))
        assigned = []
        for item in stream["increments"]:
            assigned.extend(item["train_sample_ids"])
            assigned.extend(item["probe_sample_ids"])
        self.assertCountEqual(assigned, [row["sample_id"] for row in train])

    def test_rejects_a_case_spanning_projects(self):
        train, test = synthetic_rows()
        train[-1]["case_id"] = train[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "multiple projects"):
            build_continual_stream(train, test, task=1, fold=1, seed=42)


if __name__ == "__main__":
    unittest.main()
