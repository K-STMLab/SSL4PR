import os
import argparse

def replace_in_file(file_path, old_root_path, new_root_path):
    with open(file_path, "r") as f:
        content = f.read()
    content = content.replace(old_root_path, new_root_path)
    with open(file_path, "w") as f:
        f.write(content)
        
def main(args):
    old_root_path = args.old_root_path
    new_root_path = args.new_root_path
    # for each folder in the current folder
    for folder in os.listdir():
        if os.path.isdir(folder):
            # for each file in the folder
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if file.endswith(".csv"):
                    replace_in_file(file_path, old_root_path, new_root_path)
                    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_root_path", required=True, help="Old root path")
    ap.add_argument("--new_root_path", required=True, help="New root path")
    args = ap.parse_args()
    main(args)