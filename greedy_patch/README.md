# Greedy Pixel-wise Patch Optimization for PIDNet-s

This folder contains an implementation of a greedy pixel-wise patch optimization approach for PIDNet-s. The implementation creates adversarial patches that can be applied to images to cause misclassification in the PIDNet-s semantic segmentation model.

## Overview

The greedy pixel-wise patch optimization approach works as follows:

1. Load a pre-trained PIDNet-s model and an existing patch (or create a new random one)
2. Take an image from the Cityscapes dataset
3. Apply the patch to the image
4. Create a pixel-level priority map based on the sum of absolute gradient values across all 3 channels
5. Visit each pixel in the patch in order of priority (highest to lowest)
6. For each pixel, try +2/255, 0, or -2/255 adjustments and select the one that maximizes misclassification
7. Update the patch as you go through each pixel
8. Save the optimized patch and print results

## Files

- `greedy_optimizer.py`: Core implementation of the greedy patch optimization algorithm
- `main.py`: Script to run the optimization process
- `evaluate.py`: Script to evaluate and visualize the optimized patch on test images
- `README.md`: This documentation file

## Usage

### Optimizing a Patch

To optimize a patch using the greedy pixel-wise approach:

```bash
python -m greedy_patch.main --config configs/config.yaml --num_images 10 --output_dir greedy_patch/results
```

Optional arguments:
- `--initial_patch`: Path to an initial patch to start optimization from (if not provided, a random patch will be created)
- `--num_images`: Number of images to use for optimization (default: 10)
- `--output_dir`: Directory to save results (default: greedy_patch/results)
- `--device`: Device to use (cuda:0, cpu, etc.)

### Evaluating a Patch

To evaluate an optimized patch on test images:

```bash
python -m greedy_patch.evaluate --config configs/config.yaml --patch greedy_patch/results/optimized_patch.pt --num_samples 10 --output_dir greedy_patch/evaluation
```

Required arguments:
- `--patch`: Path to the patch file to evaluate

Optional arguments:
- `--num_samples`: Number of test samples to evaluate (default: 10)
- `--output_dir`: Directory to save evaluation results (default: greedy_patch/evaluation)
- `--device`: Device to use (cuda:0, cpu, etc.)

## Implementation Details

### Priority Map

The priority map is computed based on the sum of absolute gradient values across all 3 channels. This helps identify which pixels in the patch have the most influence on the model's output.

### Pixel Adjustment

For each pixel in the patch, we try three possible adjustments: +2/255, 0, or -2/255. We select the adjustment that maximizes misclassification, which is measured by the number of incorrectly predicted pixels.

### Evaluation Metrics

The evaluation script computes several metrics:
- Pixel Accuracy: Percentage of correctly classified pixels
- Mean IoU: Mean Intersection over Union across all classes
- Misclassification Rate: Percentage of pixels that are misclassified
- IoU Reduction: Reduction in Mean IoU compared to clean images

## Results

The optimization process saves the following results:
- `optimized_patch.pt`: The optimized patch as a PyTorch tensor
- `optimized_patch.png`: Visualization of the optimized patch
- `miou_progression.png`: Plot showing the progression of Mean IoU during optimization
- `iou_metrics.npy`: Raw IoU metrics data

The evaluation process saves:
- Visualizations of test images with and without the patch
- Evaluation metrics in `evaluation_results.txt` and `evaluation_results.npy`

## References

This implementation is based on the approach described in the project requirements and uses components from the existing codebase, particularly from `trainer/trainer_TranSegPGD_AdvPatch.py`.