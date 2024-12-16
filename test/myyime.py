import json
from datetime import datetime

timedef = "/Users/deny/optimized-facerec/assets/timedef.json"

def is_time_in_range(start_time, end_time):
    # Get the current time
    current_time = datetime.now().time()
    
    # Check if current time is within the range
    if start_time <= current_time <= end_time:
        return True
    else:
        return False

# # Example usage:
# start = datetime.strptime("12:00:00", "%H:%M:%S").time()  # Start time (e.g., 8 AM)
# end = datetime.strptime("17:00:00", "%H:%M:%S").time()  # End time (e.g., 5 PM)

# print(is_time_in_range(start, end))

with open(timedef, 'r') as file:
    data = json.load(file)

for i in data:
    start = datetime.strptime(i['start'], "%H:%M:%S").time()
    end = datetime.strptime(i['end'], "%H:%M:%S").time()
    
    if is_time_in_range(start, end):
        # print(is_time_in_range(start, end))
        print(start)
        print(end)
        print(i["id"])
        
        break
# print(len(data))