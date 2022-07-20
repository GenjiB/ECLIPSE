kill $(ps aux | grep "main_task_retrieval.py" | grep -v grep | awk '{print $2}')
