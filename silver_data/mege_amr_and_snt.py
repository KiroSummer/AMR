import time


def read_snt(filepath):
    file = open(filepath, 'r')
    snts = []
    for line in file:
        line = line.strip()
        snts.append(line)
    print("read {} samples from {}".format(len(snts), filepath))
    return snts


def merge_snts_and_amrs(snts, amrs, outfilepath):
    outputfile = open(outfilepath, 'w')
    for idx, snt, amr in enumerate(zip(snts, amrs)):
        outputfile.write("# ::id " + "2m_silver_data " + str(idx) + '\n')
        outputfile.write("# ::snt " + snt + '\n')
        outputfile.write("# ::save-data " + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) + '\n')
        outputfile.write(amr + '\n')
        outputfile.write('\n')
    outputfile.close()


if __name__ == "__main__":
    snt_file_path = "/data2/qrxia/data/AMR/silver_data/2m_silver_amr/2m_0000.snt"
    amr_file_path = "/data2/qrxia/data/AMR/silver_data/2m_silver_amr/2m_0000.amr"
    snts = read_snt(snt_file_path)
    amrs = read_snt(amr_file_path)
    merge_snts_and_amrs(snts, amrs, "/data2/qrxia/data/AMR/silver_data/2m_silver_amr/2m_silver.txt")

