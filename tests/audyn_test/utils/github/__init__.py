import os
import subprocess
import warnings


def retrieve_git_branch() -> str:
    """Retrieve current branch.

    .. note::

        You may need to set environmental variable ``GITHUB_HEAD_REF`` properly.

    """
    head_ref = os.getenv("GITHUB_HEAD_REF")

    if head_ref:
        branch = head_ref
    else:
        try:
            completed_process = subprocess.run(
                "git rev-parse --abbrev-ref HEAD", shell=True, check=True, capture_output=True
            )
            branch = completed_process.stdout.decode().strip()
        except subprocess.CalledProcessError:
            branch = None
            warnings.warn("The system is not managed by git.", UserWarning, stacklevel=2)

    return branch
