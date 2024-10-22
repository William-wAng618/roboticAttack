import pickle

def check_pickle(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == '__main__':
    file_path = '/home/tw9146/tw/openvla-main/dataset/scripted_raw/2022-12-08_pnp_soft_toys/2022-12-08_17-51-04/raw/traj_group0/traj0/obs_dict.pkl'
    data = check_pickle(file_path)
    a=1