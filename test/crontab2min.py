from crontab import CronTab

# Create a new cron object for the current user
cron = CronTab(user=True)

# Define the command you want to run (e.g., running your script)
# command = 'python3 /path/to/your/script.py'
command = '/Users/deny/optimized-facerec/.venv/bin/python /Users/deny/optimized-facerec/test/test_cron.py'

# Create a new cron job with the specific command
job = cron.new(command=command)

# Set the job to run on even minutes (0, 2, 4, 6, ..., 58)
# job.minute = '0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58'
job.setall('*/2 * * * *')

reboot_job = cron.new(command=command)
reboot_job.setall('@reboot')

cron.write()
# Print confirmation
print("Cron job set to run on even minutes (0, 2, 4, 6, ..., 58).")
