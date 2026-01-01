# GitHub Repository Setup Instructions

This document provides step-by-step instructions for setting up and pushing the BEIR benchmarking repository to GitHub.

## Prerequisites

1. **Git installed**: Ensure Git is installed on your system
2. **GitHub account**: You need a GitHub account with access to the RankSaga organization
3. **Authentication**: Set up GitHub authentication (SSH keys or personal access token)

## Step 1: Initialize Git Repository

If the repository is not already a git repository:

```bash
cd benchmarking
git init
```

## Step 2: Create GitHub Repository

1. Go to [GitHub RankSaga organization](https://github.com/orgs/RankSaga/repositories)
2. Click "New repository"
3. Repository name: `beir-benchmarking`
4. Description: "RankSaga BEIR Benchmarking: Embedding Model Optimization Results"
5. Visibility: Public
6. **Do NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 3: Add All Files

```bash
# Add all files
git add .

# Check what will be committed
git status
```

## Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: RankSaga BEIR benchmarking results and code

- Complete benchmarking code and scripts
- Fine-tuning implementation with Multiple Negatives Ranking Loss
- Comprehensive results and visualizations
- Documentation (methodology, datasets, deployment)
- Model export utilities for Hugging Face
- MIT License"
```

## Step 5: Add Remote and Push

```bash
# Add remote (replace with your GitHub username/org if different)
git remote add origin https://github.com/RankSaga/beir-benchmarking.git

# Or if using SSH:
# git remote add origin git@github.com:RankSaga/beir-benchmarking.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 6: Verify Upload

1. Visit: https://github.com/RankSaga/beir-benchmarking
2. Verify all files are present
3. Check that README renders correctly
4. Verify LICENSE file is present

## Optional: Set Up Repository Settings

### Repository Topics

Add topics to help discoverability:
- `beir`
- `embeddings`
- `information-retrieval`
- `fine-tuning`
- `benchmarking`
- `sentence-transformers`
- `ranksaga`

### Repository Description

Update description if needed:
```
RankSaga BEIR Benchmarking: Comprehensive evaluation of embedding model optimization techniques. Includes code, results, visualizations, and fine-tuned models achieving up to 51% improvement on medical information retrieval.
```

### Add Collaborators

If needed, add team members as collaborators in repository settings.

## File Structure Verification

Ensure the repository contains:

```
beir-benchmarking/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config.py
├── run_baseline.py
├── fine_tune_model.py
├── run_optimized.py
├── compare_results.py
├── generate_blog_visualizations.py
├── export_model_for_hf.py
├── upload_to_hf.py
├── modal_app.py
├── modal_fine_tune_advanced.py
├── docs/
│   ├── METHODOLOGY.md
│   ├── DATASETS.md
│   └── DEPLOYMENT.md
├── examples/
│   └── quick_start.py
├── utils/
│   ├── data_loader.py
│   └── evaluation.py
└── results/
    ├── baseline/
    │   └── *.json
    ├── optimized/
    │   └── *.json
    └── domain_specific/
        └── ...
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:

**Using HTTPS**:
```bash
# Use personal access token
git remote set-url origin https://YOUR_TOKEN@github.com/RankSaga/beir-benchmarking.git
```

**Using SSH**:
```bash
# Ensure SSH key is added to GitHub account
# Test connection:
ssh -T git@github.com
```

### Large File Issues

If you encounter issues with large files:

1. Ensure `.gitignore` excludes large model files
2. Use Git LFS for large files if needed:
```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.pt"
```

### Permission Issues

If you get permission errors:
- Ensure you have write access to RankSaga organization
- Check repository settings in GitHub
- Contact organization admin if needed

## Next Steps

After successfully pushing to GitHub:

1. ✅ Verify repository is accessible
2. ✅ Add repository description and topics
3. ✅ Create releases/tags if needed
4. ✅ Share repository link in blog post and documentation
5. ✅ Monitor issues and pull requests

## Repository Maintenance

### Updating Repository

```bash
# Make changes
git add .
git commit -m "Description of changes"
git push
```

### Creating Releases

For important milestones:

```bash
# Tag a release
git tag -a v1.0.0 -m "Initial release: BEIR benchmarking results"
git push origin v1.0.0
```

Then create a release on GitHub with release notes.

## Support

If you encounter issues:
- Check GitHub documentation
- Contact repository maintainers
- Open an issue on the repository

