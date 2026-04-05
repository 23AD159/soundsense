import os
import requests
import sys

# Freesound API Config
# You need to get an API key from https://freesound.org/apiv2/apply/
API_KEY = "3y0N5ODQuzcxFoV8sWTyoHNcSmmcx8yMAJTiSZ4P"
BASE_URL = "https://freesound.org/apiv2"

def search_and_download(query, target_dir, num_results=10):
    """
    Searches Freesound for a query and downloads the top N results.
    """
    if API_KEY == "YOUR_FREESOUND_API_KEY_HERE":
        print("❌ Error: Please set your Freesound API Key in the script.")
        return

    print(f"Searching for: '{query}'...")
    
    # 1. Search
    search_url = f"{BASE_URL}/search/text/"
    params = {
        "query": query,
        "token": API_KEY,
        "fields": "id,name,previews",
        "page_size": num_results,
        # No type filter - accept any format (previews are always mp3)
    }
    
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        print(f"Failed to search: {response.text}")
        return
        
    results = response.json().get('results', [])
    print(f"Found {len(results)} matches.")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for sound in results:
        sound_id = sound['id']
        name = sound['name'].replace(" ", "_").replace("/", "_")
        preview_url = sound['previews']['preview-hq-mp3'] # Previews are often better than raw if account lacks download perms
        
        file_path = os.path.join(target_dir, f"{sound_id}_{name}.mp3")
        
        if os.path.exists(file_path):
            print(f"Skipping {name} (exists)")
            continue
            
        print(f"Downloading: {name}...")
        try:
            r = requests.get(preview_url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Error downloading {name}: {e}")

def run_scraper():
    """
    Downloads Indian-specific and domain-relevant sounds from FreeSound.
    
    NOTE: The following sounds are ALREADY in ESC-50 and don't need scraping:
      - baby_crying      → ESC-50 class: crying_baby
      - dog_barking      → ESC-50 class: dog
      - glass_breaking   → ESC-50 class: glass_breaking
      - door_knocking    → ESC-50 class: door_wood_knock
      - clapping         → ESC-50 class: clapping
      - footsteps        → ESC-50 class: footsteps
      - washing_machine  → ESC-50 class: washing_machine
    
    The following NEED FreeSound downloads (Indian-specific / missing from ESC-50):
    """
    categories = {
        # Indian-specific sounds
        "pressure_cooker_whistle": "pressure cooker whistle steam india",
        "mixer_grinder":           "mixer grinder blender india kitchen",
        "auto_rickshaw_horn":      "auto rickshaw horn india tuk tuk",
        "motorcycle_horn":         "motorcycle horn india traffic",
        "temple_bells":            "temple bell india pooja",
        "indian_electric_doorbell":"indian doorbell electric ring",
        "traditional_bell":        "traditional bell doorbell ding dong",
        "phone_ringtone_indian":   "indian phone ringtone mobile",
        "lpg_gas_alarm":           "gas leak alarm LPG beep",
        "fire_alarm":              "fire alarm smoke detector siren",
        "tv_background":           "television background indoor ambient",
        "washing_machine_beep":    "washing machine beep done end cycle",
        "name_calling":            "person calling name hey shouting",
        # Already in ESC-50 but augmenting with more real-world samples
        "pressure_cooker_hiss":    "pressure cooker hiss steam release",
    }
    
    print(f"\n{'='*50}")
    print(f"Starting FreeSound scraper for {len(categories)} categories...")
    print(f"{'='*50}\n")
    
    for folder, query in categories.items():
        target_path = os.path.join("data/raw", folder)
        print(f"\n--- Category: {folder} ---")
        search_and_download(query, target_path, num_results=20)
    
    print(f"\n{'='*50}")
    print("✅ Scraping complete! Check data/raw/ for downloaded files.")
    print("Next step: Run src/processing/prepare_data.py to build the training dataset.")
    print(f"{'='*50}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    run_scraper()
