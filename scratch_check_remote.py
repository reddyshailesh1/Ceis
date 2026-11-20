import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('172.16.72.6', username='sreekanthreddy', password='mech@123', timeout=10)

test_script = '''
from laya.train import load_checkpoint
import torch
print("Loading checkpoint on remote RTX 6000 Ada...")
model, tok, cfg = load_checkpoint("pavanmantha/mednli_laya")
print("Successfully loaded model:", type(model))
print("Moving to cuda...")
model = model.to("cuda")
print("Model moved to cuda successfully! Memory:", round(torch.cuda.memory_allocated() / (1024**2), 2), "MB")
'''

stdin, stdout, stderr = ssh.exec_command(f"/home/sreekanthreddy/gemma4-env/bin/python -c '{test_script}' 2>&1")
print(stdout.read().decode('utf-8', errors='replace').strip())
print(stderr.read().decode('utf-8', errors='replace').strip())

ssh.close()
