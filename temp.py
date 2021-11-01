from pathlib import Path

path1 = Path("/c/folder/subfolder/")
path2 = Path("/C/yfiletxt/")
print(path1.parent)
print(path2.parent)

def is_valid_text(class_dict, text_data):
    for ch in text_data:
        ch = ch.upper()
        if ch == ' ':
            continue
        if class_dict.get(ch) is None:
            print(f'{ch} is not found')
            return False
    return True


if __name__ == '__main__':
    text = '수심 32m'
    class_dict = {'수':1, '심':1, '3':1, '2': 1, 'M': 1}
    is_valid_text(class_dict, text)
