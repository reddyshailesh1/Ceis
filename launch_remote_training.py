import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('172.16.72.6', username='sreekanthreddy', password='mech@123', timeout=10)

cmd = '''
mkdir -p /home/sreekanthreddy/ceis/runs
cd /home/sreekanthreddy/ceis
nohup /home/sreekanthreddy/gemma4-env/bin/python main.py train --config config/train_config.yaml --out runs/ceis_mednli_v1 --device cuda:0 > /home/sreekanthreddy/ceis/runs/train_remote.log 2>&1 &
echo $! > /home/sreekanthreddy/ceis/runs/train.pid
echo "PID: $(cat /home/sreekanthreddy/ceis/runs/train.pid)"
'''

stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode('utf-8', errors='replace').strip())
print(stderr.read().decode('utf-8', errors='replace').strip())

ssh.close()
