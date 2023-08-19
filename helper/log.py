import os, sys

def write_log(log_dir, runname, log):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, runname + '.log'), 'a') as f:
        f.write(log + '\n')
    print(log)
    sys.stdout.flush()