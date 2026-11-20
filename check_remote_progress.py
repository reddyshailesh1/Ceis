import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('172.16.72.6', username='sreekanthreddy', password='mech@123', timeout=10)

cmd = '''
ps -p $(cat /home/sreekanthreddy/ceis/runs/train.pid) -o pid,stat,%cpu,%mem,etime
tail -n 25 /home/sreekanthreddy/ceis/runs/train_remote.log
'''
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode('utf-8', errors='replace').strip())
ssh.close()
