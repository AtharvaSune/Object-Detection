from train import train
from test import test
import  argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom object detection")
    parser.add_argument("-g", "--gpu", type=bool, required= True,
                        help="True if GPU available else False")
    sub_parser = parser.add_subparsers(title="subcommands",help="sub command help", dest="subcommands")

    train_parser = sub_parser.add_parser('train', help="train help")
    train_parser.add_argument('-r', "--root", type=str, required=True,
                        help="root path to the dataset folder")
    train_parser.add_argument('-n', "--num_classes", type=int, required=True, 
                        help="number of classes")
    train_parser.add_argument('-e', "--epochs", type=int, required=True, 
                        help="number of epochs")
    train_parser.add_argument('-b', "--batch_size", type=int, 
                        help="batch size for training (power of 2)")

    test_parser = sub_parser.add_parser(test, help="test help")
    test_parser.add_argument('-i', "--img_path", type=str, required=True,
                        help="test image path")
    test_parser.add_argument('-f', "--file_path", type=str, required=True,
                        help="output path for the processed image")
    test_parser.add_argument('-n', "--num_classes", type=int, required=True, 
                        help="number of classes")
    test_parser.add_argument("-m", "--model_path", type=str, required=True,
                            help="Path to model weights")

    args = parser.parse_args()

    try:
        if args.subcommands == "train":
            train(args.root, args.epochs, args.num_classes, args.gpu)
        elif args.subcommands == 'test':
            test(args.img_path, args.file_path, args.num_classes, args.model_path, args.gpu)
    except Exception as e:
        print(e)