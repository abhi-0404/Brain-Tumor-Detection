# Brain Tumor MRI Dataset

This dataset contains a large amount of MRI images related to brain tumors. Due to its size, we cannot upload it directly here. However, you can download the dataset from the following link:

[Download Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Please download the dataset and place it in this directory before running any scripts or notebooks that require it.

## Folder Notes

- Keep the raw training and testing data outside the repository if the dataset is too large for Git.
- The Python package, CLI, notebook, and web app expect this folder layout.
- Do not rename class folders unless you update the notebook labels and preprocessing paths.

## Expected Local Layout

Use this local structure after downloading from Kaggle:

```
Dataset/
	Training/
		glioma/
		meningioma/
		notumor/
		pituitary/
	Testing/
		glioma/
		meningioma/
		notumor/
		pituitary/
```
