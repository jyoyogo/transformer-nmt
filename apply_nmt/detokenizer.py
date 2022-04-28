import sys

if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            if ' ' in line:
                pass
                # line = line.strip().replace(' ', '').replace('_', ' ').strip()
            else:
                # line = line.strip().replace('#', '').strip()
                pass

            sys.stdout.write(line + '\n')
        else:
            sys.stdout.write('\n')
