import pickle
data = pickle.load(open('data/nuscenes/nuscenes_infos_train.pkl', 'rb'))
items = data.get('data_list', data.get('infos', []))
print("Keys:", list(items[0].keys()))
print("sample[0] scene_description:", items[0].get('scene_description', 'N/A'))
print("sample[0] scene_token:", items[0].get('scene_token', 'N/A'))
