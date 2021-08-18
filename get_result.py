import statistics 
import numpy as np
import glob

if __name__ == '__main__':
    path = './result/'
    files = glob.glob(path + '*.res')
    k = 0

    fwrite = open('result.out', 'w')
    for filename in sorted(files):
        bacc_list = []
        for line in open(filename, 'r').readlines():
            _, acc, bacc = line.strip().split('\t')
            bacc_list.append(float(bacc))

            if k > 0 and len(bacc_list) > k: 
                bacc_list = bacc_list[1:]
        fwrite.write('%s\t%s\t%s\t%s\n' % (filename, str(len(bacc_list)),str(np.std(bacc_list)), str(sum(bacc_list) / len(bacc_list))))
    fwrite.close()
        


