"""Evaluate context-only search with human-labeled query/video-ID pairs.
CSV columns: query,expected_id. One row per relevant video per query.
"""
import argparse
import csv
from collections import defaultdict
from kitty_ai import find_media


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_csv', help='CSV with query,expected_id columns')
    parser.add_argument('--output', default='evaluation_results.csv')
    args = parser.parse_args()
    expected = defaultdict(set)
    with open(args.labels_csv, newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            expected[row['query']].add(int(row['expected_id']))
    if not expected:
        raise SystemExit('No labeled queries found')
    details = []
    for query, relevant in expected.items():
        results = find_media(query, limit=5)
        ids = [item['id'] for item in results]
        details.append({'query': query, 'expected_ids': ';'.join(map(str, sorted(relevant))),
                        'returned_ids': ';'.join(map(str, ids)),
                        'top1': int(bool(ids) and ids[0] in relevant),
                        'precision_at_3': sum(item in relevant for item in ids[:3]) / 3,
                        'hit_at_3': int(any(item in relevant for item in ids[:3]))})
    with open(args.output, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    n = len(details)
    print(f'Queries: {n}; Top-1: {sum(x["top1"] for x in details)/n:.3f}; '
          f'Hit@3: {sum(x["hit_at_3"] for x in details)/n:.3f}; '
          f'Precision@3: {sum(x["precision_at_3"] for x in details)/n:.3f}')
    print('Note: existing bot interface returns at most 3, so Hit@5 is not measurable here.')


if __name__ == '__main__':
    main()
