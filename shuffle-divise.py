import os
import random
import shutil

input_dir = 'lion_tiger/train/tigers/'
output_dir = 'lion_tiger/test/tigers/'

percentage = 0.2
li = os.listdir(input_dir)
n = len(li)
p = int(percentage * n)
s = set()

# Generate random indices to select files for moving
for i in range(p):
    s.add(random.randint(0, n-1))

# Move selected files from input_dir to output_dir
for i in s:
    file_to_move = li[i]
    source_path = os.path.join(input_dir, file_to_move)
    destination_path = os.path.join(output_dir, file_to_move)
    shutil.move(source_path, destination_path)