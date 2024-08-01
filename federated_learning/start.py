import subprocess
import time
# Specifica il percorso del programma Python che vuoi eseguire
percorso_programma = "/home/aleleo/tesi_code/code_python/federated_learning/server.py"

# Apre un nuovo terminale e esegue il programma specificato
subprocess.Popen(["gnome-terminal", "--", "python3", percorso_programma])

percorso_login='/home/aleleo/tesi_code/code_python/login_ssi/login.py'

subprocess.Popen(["gnome-terminal", "--", "python3", percorso_login])






percorso_programma = "/home/aleleo/tesi_code/code_python/federated_learning/client_fed.py"
i=0
for i in range(4):
    subprocess.Popen(["gnome-terminal", "--", "python", percorso_programma,"--partition-id",str(i),"--port",str(6001+i)])
    
directory = "/home/aleleo/tesi_code/code_python/frontend/frontend/src"

subprocess.run(["npm", "start"], cwd=directory)
