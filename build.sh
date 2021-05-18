module load sge

qrsh -q arcus.q -P arcus_gpu.p -l hostname=arcus-4.ics.uci.edu -l gpu=1  -pty y $SHELL

module load sge

qstat -q arcus.q -F gpu -u "*"

#python main.py
