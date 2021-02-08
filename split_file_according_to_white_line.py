import sys
import itertools


def permutation_with_repeats(seq, key):
    """
    generator that produces all permutations of length key
    of the elements in  seq.
    seq = list('abc'); key = 4
    >>> aaaa aaab aaac aaba aabb aabc aaca aacb...
    seq = list('abcdefghijklmnopqrstuvwzyz'); key = 2
    >>> aa ab ac ad ae af ag ah ai aj ak al...
    """
    for _ in  itertools.product(seq, repeat=key):
        yield ''.join(_)


seq = list('abcdefghijklmnopqrstuvwzyz')
key = 2
a = permutation_with_repeats(seq, key)


def write_to_new_file(instances):
    output_file_name = next(a)
    output_file = open(output_file_name, 'w')
    # for line in instances:
    #     output_file.write(line)
    output_file.close()


if __name__ == "__main__":
    instance_number = int(sys.argv[2])
    file_name = sys.argv[1]

    instance_count = 0
    input_file = open(file_name, 'r')
    instances = []
    for line in input_file:
        instances.append(line)
        if line.strip() == '':
            instance_count += 1
            if instance_count >= instance_number:
                write_to_new_file(instances)
                instance_count = 0
                instances = []
        else:
            pass
    write_to_new_file(instances)
    instance_count = 0
    instances = []

