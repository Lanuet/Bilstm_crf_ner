import argparse
import initial
import ner

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="train file path")
parser.add_argument("--dev_dir", help="dev file path")
parser.add_argument("--test_dir", help="test_file path")
args = parser.parse_args()

train_dir = args.train_dir
dev_dir = args.dev_dir
test_dir = args.test_dir

def main():
    initial.main(train_dir,dev_dir,test_dir)
    ner.main(train_dir,dev_dir, test_dir)
if __name__ == '__main__':
    main()