import pickle

def check_pkl():
    with open('data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Keys in root:", data.keys() if isinstance(data, dict) else "Not a dict")
    
    if isinstance(data, dict):
        infos = data.get('data_list', data.get('infos', []))
    else:
        infos = data
        
    print(f"Loaded {len(infos)} samples.")
    
    sample = infos[10]
    print("Keys in sample 10 radars['RADAR_FRONT']:", sample['radars']['RADAR_FRONT'].keys())
    
    if 'radar_sweeps' in sample['radars']['RADAR_FRONT']:
        sweeps = sample['radars']['RADAR_FRONT']['radar_sweeps']
        print(f"Success! Found `radar_sweeps` with {len(sweeps)} sweeps.")
    else:
        print("Failed to find `radar_sweeps` in the radars dict.")

check_pkl()