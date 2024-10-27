def download_annotations(
    type: str, subset: Optional[Union[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    if subset is None or subset == "all":
        subset = ["train", "validation", "test"]

    if isinstance(subset, str):
        subset = [subset]

    annotations = []

    if type == "won":
        for _subset in subset:
            url = f"https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/mtat_{type}_{_subset}.jsonl"
            download_file_from_github_release(url, path)

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    annotation = json.loads(line)
                    annotations.append(annotation)
