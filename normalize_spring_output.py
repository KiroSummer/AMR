import sys


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = input_file_path + ".normalize"

    with open(input_file_path, 'r') as f:
        with open(output_file_path, 'w') as o:
            for line in f:
                if line.startswith("# ::nsent"):
                    o.write("# ::id " + line[9:])
                else:
                    o.write(line)
