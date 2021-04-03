import sys
import os
import subprocess


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isdir(input_dir) is False:
        print("argv 1 should be a dir")
        sys.exit(-1)
    if os.path.isdir(output_dir) is False:
        print("argv 2 should be a dir")
        sys.exit(-1)
    for file in os.listdir(input_dir):
        file_prefix = file.split('.')[0]
        if os.path.exists(os.path.join(input_dir, file_prefix + '.pred')):  # parse done
            parsed_done_files = [os.path.join(input_dir, file_prefix),
                                 os.path.join(input_dir, file_prefix + '.pred'),
                                 os.path.join(input_dir, file_prefix + '.gold')]
            for f in parsed_done_files:
                child = subprocess.Popen('{} {} {}'.format("mv", f, output_dir), shell=True)
                child.wait()
        else:
            if os.path.exists(os.path.join(input_dir, file_prefix + '.gold')):
                child = subprocess.Popen('{} {}'.format("rm", os.path.join(input_dir, file_prefix + '.gold')), shell=True)
                child.wait()

