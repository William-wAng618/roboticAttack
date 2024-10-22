import pickle

def check_pickle(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == '__main__':
    file_path = '/Volumes/Extreme Pro/dataset/OpenX-Embodiment/asu_table_top_converted_externally_to_rlds/asu_table_top_converted_externally_to_rlds_00000/sample_000000000000.data.pickle'
    data = check_pickle(file_path)
    a=1
    import numpy as np
    # import cv2
    from PIL import Image
    import io
    image_binary = data["steps"][0]["observation"]["image"]
    image = Image.open(io.BytesIO(image_binary))

    # 显示图像
    # image.show()
    image.save('test.png')
    # language instruction
    # "put down blue can"