import os

classes = [
    'pressure_cooker', 'indian_doorbell_electric', 'traditional_doorbell', 'baby_crying', 
    'auto_rickshaw_horn', 'motorcycle_horn', 'fire_alarm', 'lpg_leak', 
    'dog_barking', 'phone_ringing', 'mixer_grinder', 'cooker_whistle', 
    'temple_bells', 'name_calling', 'glass_breaking', 'knocking', 
    'clapping', 'footsteps', 'tv_sounds', 'washing_machine'
]

base_dir = "data/raw"

for cls in classes:
    path = os.path.join(base_dir, cls)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")

print("✅ Folders ready.")
