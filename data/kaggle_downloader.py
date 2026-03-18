"""
Kaggle dataset downloader utility.
"""
import os
import subprocess
import shutil


KAGGLE_DATASETS = {
    'heart': {
        'slug': 'sintariosatya/heart-disease-dataset',
        'filename': 'heart.csv',
        'search_names': ['heart.csv', 'heart_disease.csv']
    },
    'student': {
        'slug': 'nabeelqureshitiii/student-performance-dataset',
        'filename': 'student_performance.csv',
        'search_names': ['student_performance.csv', 'student_data.csv', 'Student_Performance.csv']
    },
    'housing': {
        'slug': 'yasserh/housing-prices-dataset',
        'filename': 'housing.csv',
        'search_names': ['Housing.csv', 'housing.csv', 'house_prices.csv']
    }
}


def check_kaggle_credentials():
    """Check if Kaggle API credentials are available."""
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    env_username = os.environ.get('KAGGLE_USERNAME')
    env_key = os.environ.get('KAGGLE_KEY')
    return os.path.exists(kaggle_json) or (env_username and env_key)


def download_dataset(slug, target_filename, raw_dir='./data/raw/'):
    """
    Download a Kaggle dataset.

    Args:
        slug: Kaggle dataset slug (owner/dataset-name)
        target_filename: desired filename after download
        raw_dir: directory to save files

    Returns:
        bool: True if successful
    """
    os.makedirs(raw_dir, exist_ok=True)

    try:
        import kaggle
        dataset_name = slug.split('/')[-1]

        # Download to temp dir
        temp_dir = os.path.join(raw_dir, '_temp_download')
        os.makedirs(temp_dir, exist_ok=True)

        kaggle.api.dataset_download_files(
            slug, path=temp_dir, unzip=True, quiet=False
        )

        # Find and move the CSV file
        target_path = os.path.join(raw_dir, target_filename)

        # Search for CSV in download dir
        dataset_info = None
        for name, info in KAGGLE_DATASETS.items():
            if info['slug'] == slug:
                dataset_info = info
                break

        search_names = dataset_info['search_names'] if dataset_info else [target_filename]

        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                if fname.lower().endswith('.csv'):
                    for search_name in search_names:
                        if fname.lower() == search_name.lower():
                            shutil.copy(os.path.join(root, fname), target_path)
                            print(f"  Saved: {target_path}")
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return True

        # If not found by exact name, take first CSV
        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                if fname.lower().endswith('.csv'):
                    shutil.copy(os.path.join(root, fname), target_path)
                    print(f"  Saved (first CSV found): {target_path}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return True

        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"  No CSV found in download for {slug}")
        return False

    except Exception as e:
        print(f"  Download failed for {slug}: {e}")
        return False


def download_all_datasets(raw_dir='./data/raw/'):
    """Download all 3 Kaggle datasets."""
    results = {}
    for name, info in KAGGLE_DATASETS.items():
        print(f"\nDownloading {name} dataset ({info['slug']})...")
        success = download_dataset(info['slug'], info['filename'], raw_dir)
        results[name] = success
    return results


def check_dataset_availability(raw_dir='./data/raw/'):
    """Check which datasets are available locally."""
    status = {}
    for name, info in KAGGLE_DATASETS.items():
        path = os.path.join(raw_dir, info['filename'])
        status[name] = os.path.exists(path)
    return status
