from pathlib import Path

from inspect_ai import Task, eval, task
from inspect_ai.agent import human_cli
from inspect_ai.scorer import includes
from inspect_cyber import create_agentic_eval_dataset


@task
def doomla():
    return Task(
        dataset=(
            create_agentic_eval_dataset(
                root_dir=Path("evals/doomla").resolve()
            ).filter_by_metadata({"variant_name": "example"})
        ),
        solver=human_cli(),
        scorer=includes(),
    )


eval(
    doomla,
    # model="openai/o4-mini",
    # max_tokens=5000,
    # epochs=3,
)
