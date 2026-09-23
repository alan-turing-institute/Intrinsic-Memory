"""The result files' schemas agree with each other.

Every file names the same identity columns first and `seed` last, and the token
counts appear in the same order wherever they appear, so a reader can line up a
task row, a progress row and an overall_results row. Nothing in the writers
would notice the schemas drifting apart.
"""

from tasks import results

SCHEMAS = {
    'the task results file': results.TASK_COLUMNS,
    'the progress and overall_results files': results.AGGREGATE_COLUMNS,
    'failed_tasks.csv': results.FAILED_TASK_COLUMNS,
    'failed_experiments.csv': results.FAILED_EXPERIMENT_COLUMNS,
}

CARRY_TOKENS = ('the task results file', 'the progress and overall_results files')


def test_every_result_schema_is_aligned():
    for name, columns in SCHEMAS.items():
        assert columns[:len(results.IDENTITY_COLUMNS)] == results.IDENTITY_COLUMNS, (
            f"{name} does not open with the identity columns"
        )
        assert columns[-1] == 'seed', f"{name} does not end in seed"
        assert len(set(columns)) == len(columns), f"{name} names a column twice"

    block = results.SPEND_COLUMNS
    for name in CARRY_TOKENS:
        columns = SCHEMAS[name]
        start = columns.index(block[0])
        assert columns[start:start + len(block)] == block, (
            f"{name} does not carry the token columns in the shared order"
        )
