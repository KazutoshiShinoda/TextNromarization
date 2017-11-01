import argparse
parser = argparse.ArgumentParser(description='Show some lines from a large file')
parser.add_argument('N', help='number of lines to show', type=int)
parser.add_argument('START', help='where to start', type=int)
args = parser.parse_args()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

files = ['en_train.csv']
n = args.N
start = args.START
is_chd = False
for file in files:
    print('------------------------------------------------')
    print()
    train = open("../input/" + file, encoding='UTF8')
    line = train.readline()
    i=0
    before = ''
    after = ''
    tmp_id = '0'
    while i < start + n:
        line = train.readline().strip()
        line = line.split(',')
        s_id = line[0]
        if  tmp_id == s_id:
            b = line[3][1:-1]
            a = line[4][1:-1]
            if a == b:
                before += b + ' '
                after += a + ' '
            else:
                is_chd = True
                before += bcolors.WARNING + b + bcolors.ENDC + ' '
                after += bcolors.WARNING + a +bcolors.ENDC + ' '
        else:
            if is_chd:
                if i >= start:
                    print(before)
                    print(after)
                    print()
                is_chd = False
                i+=1
            before=''
            after=''
            before += line[3][1:-1] + ' '
            after += line[4][1:-1] + ' '
            tmp_id = s_id
    print('------------------------------------------------')
