# Guide to Using Git LFS for the IPL Dataset

This project uses Git Large File Storage (LFS) to manage the large `IPL.csv` dataset. Because the dataset is too large to be stored directly in the Git repository, you will need to use Git LFS to download and manage it properly. Follow the steps below to set up Git LFS and access the dataset.

## Step 1: Install Git LFS

First, you need to install Git LFS on your local machine. The installation process varies depending on your operating system.

### macOS

If you use Homebrew, you can install Git LFS with the following command:
```bash
brew install git-lfs
```

### Windows

You can download and run the Git LFS installer from the official website:
[https://git-lfs.github.com/](https://git-lfs.github.com/)

### Linux

For Debian-based distributions like Ubuntu, you can use `apt`:
```bash
sudo apt-get install git-lfs
```
For other distributions, please refer to the official Git LFS documentation.

## Step 2: Set Up Git LFS

Once Git LFS is installed, you need to set it up for your user account. Run the following command in your terminal:
```bash
git lfs install
```
This command only needs to be run once per user account.

## Step 3: Clone the Repository

Now, you can clone the project repository as you normally would:
```bash
git clone https://github.com/your-username/ipl-impact-intelligence.git
cd ipl-impact-intelligence
```

## Step 4: Pull the LFS Files

When you clone a repository that uses Git LFS, the large files are downloaded as small "pointer" files. To download the actual `IPL.csv` dataset, run the following command from the root of the repository:
```bash
git lfs pull
```
This will download the full dataset and replace the pointer file with the actual `IPL.csv` file.

## How to Verify

After running `git lfs pull`, you can check the size of the `IPL.csv` file. It should be significantly larger than a few kilobytes. You can use the `ls -lh` command to check the file size:
```bash
ls -lh IPL.csv
```
If the file is large, you have successfully downloaded the dataset. You can now run the project as described in the `README.md` file.