import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.registry import register_aggregation, register_filter, register_metric
from lm_eval.tasks import TaskManager


def get_new_manager(load_existing: bool = True, override_config: Optional[Dict] = None):
    configs = list(
        Path(__file__).parents[1].joinpath("lm_eval_datasets").absolute().glob("*")
    )
    task_manager = TaskManager(
        include_path=configs,
        include_defaults=load_existing,
    )

    return task_manager


def load_tasks_with_overrides(
    task_names: Union[str, List[str]],
    global_overrides: Optional[Dict[str, Any]] = None,
    task_specific_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    task_manager: Optional[TaskManager] = None,
) -> Dict:
    """
    Load tasks with both global and task-specific configuration overrides.

    Args:
        task_names: Single task name or list of task names to load
        global_overrides: Configuration to apply to all tasks
        task_specific_overrides: Dictionary mapping task names to their specific overrides
        task_manager: Optional TaskManager instance (will create one if not provided)

    Returns:
        Dictionary of task objects with overrides applied
    """
    # Initialize parameters
    if isinstance(task_names, str):
        task_names = [task_names]

    if global_overrides is None:
        global_overrides = {}

    if task_specific_overrides is None:
        task_specific_overrides = {}

    if task_manager is None:
        task_manager = get_new_manager()

    # Create task configs with overrides
    task_configs = []

    for task_name in task_names:
        # Start with global overrides
        task_overrides = global_overrides.copy()

        # Apply task-specific overrides (these take precedence)
        if task_name in task_specific_overrides:
            task_overrides.update(task_specific_overrides[task_name])

        # If we have any overrides, create a config dict
        if task_overrides:
            task_config = {"task": task_name}
            task_config.update(task_overrides)
            task_configs.append(task_config)
        else:
            # No overrides, just use the task name
            task_configs.append(task_name)
    return task_configs


@register_metric(
    metric="group_exact_match",
    higher_is_better=True,
    output_type="float",
    aggregation="mean",
)
def group_exact_match(items, groups, ignore_case=False, ignore_punctuation=False):
    results = {}
    for item in items:
        group_value = item.get(groups, "unknown")
        if group_value not in results:
            results[group_value] = []

        prediction = item.get("prediction", "")
        target = item.get("target", "")

        if ignore_case:
            prediction = prediction.lower()
            target = target.lower()

        if ignore_punctuation:
            prediction = re.sub(r"[^\w\s]", "", prediction)
            target = re.sub(r"[^\w\s]", "", target)

        match = prediction.strip() == target.strip()
        results[group_value].append(float(match))

    for key in results:
        results[key] = sum(results[key]) / len(results[key]) if results[key] else 0.0

    return results


@register_metric(
    metric="multi_target_match1",
    higher_is_better=True,
    output_type="float",
    aggregation="mean",
)
def multi_target_match1(pred, doc):
    """Check if the prediction matches any target in all_targets"""
    prediction = pred.strip().lower()
    all_targets = doc.get("all_targets", [doc.get("target", "")])

    for target in all_targets:
        if prediction == target.strip().lower():
            return 1.0

    return 0.0


# Instead of a custom filter function, let's create a custom metric that handles multiple targets


@register_metric(
    metric="multi_target_match",
    higher_is_better=True,
    output_type="float",
    aggregation="mean",
)
def multi_target_match(predictions, references, **kwargs):
    """Check if prediction matches any target in all_targets"""
    print(predictions)
    print(references)
    normalized_targets = [target.strip().lower() for target in references]
    results = []
    for pred in predictions:
        # Check if prediction matches any target
        match = any([target in pred.strip().lower() for target in normalized_targets])
        results.append(float(match))
    return sum(results) / len(results) if results else 0.0
