"""This script just reconstruct unified predictions tsvs from fine and coarse tsvs"""

import os
from transformers_baseline.evaluation import evaluate_iob_files

# Set directories and paths
bsl_dir = '/scratch/sven/hipe_baseline/'
preds_path = 'results/hipe_eval/predictions.tsv'
output_dir = '/scratch/sven/hipe_baseline/formatted_results'
hipe_script_path = '/scratch/sven/packages/HIPE-scorer/clef_evaluation.py'

# Loop over each dir
for fine_dir in [d for d in os.listdir(bsl_dir) if d.endswith('_fine')]:

    # Get the corresponding coarse dir
    coarse_dir = fine_dir.replace('_fine', '_coarse')

    # Open the predictions tsvs
    with open(os.path.join(bsl_dir, fine_dir, preds_path), 'r') as f:
        fine_lines = f.read().split('\n')

    with open(os.path.join(bsl_dir, coarse_dir, preds_path), 'r') as f:
        coarse_lines = f.read().split('\n')
    assert len(fine_lines) == len(coarse_lines)

    # Merge the predictions tsv
    merged_lines = [fine_lines[0]]
    for fine_line, coarse_line in zip(fine_lines[1:], coarse_lines[1:]):
        if fine_line.startswith('#') or fine_line=='':
            assert fine_line == coarse_line
            merged_lines.append(fine_line)

        else:
            fine_line = fine_line.split('\t')
            coarse_line = coarse_line.split('\t')
            assert fine_line[0] == coarse_line[0]
            fine_line[1] = coarse_line[1]
            merged_lines.append('\t'.join(fine_line))

    # Define the HIPE-compliant bundle name and create its repo
    bundle_dir = 'neurbsl_bundle3_' + fine_dir.replace('_fine', '') + '_1'
    os.makedirs(os.path.join(output_dir, bundle_dir), exist_ok=True)

    # Write the the merged lines
    output_path = os.path.join(output_dir, bundle_dir, bundle_dir + '.tsv')
    with open(output_path, 'w') as f:
        f.write('\n'.join(merged_lines))

    # Run HIPE-evaluation
    evaluate_iob_files(output_dir=os.path.join(output_dir, bundle_dir),
                       groundtruth_path=os.path.join(bsl_dir,fine_dir, 'groundtruth.tsv'),
                       preds_path=os.path.join(output_path),
                       method='hipe',
                       hipe_script_path=hipe_script_path)
    



