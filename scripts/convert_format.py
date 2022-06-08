"""
there are some wired problems in the original AMR data, for example
# ::snt That<92>s what we<92>re with<85>You<92>re not sittin<92> there in a back alley and sayin<92> hey what do you say, five bucks?
"""
import sys


if __name__ == "__main__":
    file_path = sys.argv[1]
    with open(file_path, 'r') as input_file:
        output_file = open(file_path + '.fixencoding.txt', 'w')
        for line in input_file.readlines():
            line = str.encode(line)
            line = line.decode('utf8').encode('latin1').decode('cp1252')
            output_file.write(line)
        output_file.close()

