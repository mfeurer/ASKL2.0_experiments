import openml

def load_from_openml(task_ids):
    tasks = openml.tasks.list_tasks(task_type_id=1, output_format='dataframe')
    tasks = tasks.loc[task_ids]
    task_ids_sorted_by_num_features = sorted(task_ids,
                                             key=lambda tid: tasks.loc[tid, 'NumberOfInstances'])
    print("Loaded tasks from openml")
    return tasks, task_ids_sorted_by_num_features
