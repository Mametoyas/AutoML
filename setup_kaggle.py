"""
Kaggle setup and dataset download utility.

Usage:
  python setup_kaggle.py --check          # check status only
  python setup_kaggle.py --download       # download all 3 Kaggle datasets
  python setup_kaggle.py --use-synthetic  # create .use_synthetic flag
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Kaggle Dataset Setup')
    parser.add_argument('--check', action='store_true', help='Check Kaggle credentials and dataset availability')
    parser.add_argument('--download', action='store_true', help='Download all Kaggle datasets')
    parser.add_argument('--use-synthetic', action='store_true', help='Create flag to use synthetic data')
    args = parser.parse_args()

    raw_dir = './data/raw/'
    os.makedirs(raw_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.kaggle_downloader import (check_kaggle_credentials,
                                         download_all_datasets,
                                         check_dataset_availability,
                                         KAGGLE_DATASETS)

    print("\n" + "="*60)
    print("AutoML Metaheuristic — Kaggle Setup")
    print("="*60)

    if args.use_synthetic:
        flag_path = os.path.join(raw_dir, '.use_synthetic')
        with open(flag_path, 'w') as f:
            f.write('1')
        print(f"\n[OK] Synthetic data flag created at {flag_path}")
        print("   Run: python main.py --use-synthetic")
        return

    # Check credentials
    has_creds = check_kaggle_credentials()
    print(f"\nKaggle Credentials: {'[OK] Found' if has_creds else '[X] Not found'}")
    if not has_creds:
        print("  To set up Kaggle credentials:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Place kaggle.json in ~/.kaggle/kaggle.json")
        print("  Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")

    # Check existing datasets
    print("\nDataset Availability:")
    status = check_dataset_availability(raw_dir)
    for name, available in status.items():
        info = KAGGLE_DATASETS[name]
        icon = '[OK]' if available else '[X]'
        print(f"  {icon} {name:10s} — {info['filename']}")

    if args.download:
        if not has_creds:
            print("\n[X] Cannot download: No Kaggle credentials found.")
            print("   Use --use-synthetic to run with fallback datasets.")
            return

        print("\nDownloading datasets...")
        results = download_all_datasets(raw_dir)

        print("\nDownload Results:")
        for name, success in results.items():
            icon = '[OK]' if success else '[X]'
            print(f"  {icon} {name}")

    elif not args.check:
        parser.print_help()

    print()


if __name__ == '__main__':
    main()
