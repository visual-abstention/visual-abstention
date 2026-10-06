"""Download the full data from Hugging Face into the folders that eval/ and vista/ read.

  python data/download.py dod      # Draw-or-Decline   -> data/dod/   (2,100 requests with images)
  python data/download.py train    # VisTA training set -> data/train/ (76,448 examples with images)

DoD (visual-abstention/Draw-or-Decline) is placed as
    data/dod/instruction.jsonl, data/dod/ground_truth_criteria.jsonl, data/dod/image/*.png
and is used with  `python eval/infer.py ... --data ../data/dod`  and  `python eval/judge.py ... --data ../data/dod`.

The training set (visual-abstention/VisTA-Train) is placed as
    data/train/<category>/instruction.jsonl, data/train/<category>/ground_truth_criteria.jsonl, data/train/image/*.png
and is used with  `VISTA_DATA=../data/train bash vista/launch.sh`. Its images come as tar shards (images/*.tar,
listed with their SHA-256 in images/manifest.json); they are extracted into data/train/image/ and then deleted
unless --keep-archives is given. The 30 shards hold 98 GB, and the extracted images take about as much; because
every shard is deleted right after it is extracted, about 100 GB of disk suffice (about 200 GB with --keep-archives).
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOS = {'dod': ('visual-abstention/Draw-or-Decline', HERE / 'dod'),
         'train': ('visual-abstention/VisTA-Train', HERE / 'train')}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(folder, verify, keep):
    manifest = json.loads((folder / 'images/manifest.json').read_text())
    for shard in manifest['shards']:
        path = folder / 'images' / shard['file']
        if not path.exists():
            print('already extracted:', shard['file'])
            continue
        if verify and sha256(path) != shard['sha256']:
            raise SystemExit(f'Checksum mismatch: {path}')
        print('extracting', shard['file'], flush=True)
        with tarfile.open(path) as archive:
            # Members are image/<name>.png (and hard links to them within the same shard).
            if hasattr(tarfile, 'data_filter'):
                archive.extractall(folder, filter='data')
            else:
                archive.extractall(folder)
        if not keep:
            path.unlink()
    print(len(list((folder / 'image').iterdir())), 'images in', folder / 'image')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('which', choices=sorted(REPOS))
    parser.add_argument('--revision', help='a branch, tag or commit of the dataset repository (default: main)')
    parser.add_argument('--verify', action='store_true', help='check the SHA-256 of every image shard first')
    parser.add_argument('--keep-archives', action='store_true', help='keep the image shards after extraction')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    repo, folder = REPOS[args.which]
    snapshot_download(repo, repo_type='dataset', revision=args.revision, local_dir=folder)
    if args.which == 'train':
        extract(folder, args.verify, args.keep_archives)
    print('done:', folder)


if __name__ == '__main__':
    main()
