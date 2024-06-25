# Cloning your repo
1. ```bash
   git clone https://github.com/EduCollaborations/Calculation-of-Object-Dimensions
   ```
2. once this done redirect the directory into the project
  ```bash
   cd Calculation-of-Object-Dimensions
   ```
Creating a new branch is a common task in version control systems like Git. Here's how you can create a new branch, switch to it, and push it to a remote repository like GitHub.

### Creating a New Branch Locally

1. **Open your terminal or command prompt.**
2. **Navigate to your Git repository** if you are not already in it:

   ```bash
   cd /path/to/your/repository
   ```

3. **Create a new branch** using the `git branch` command followed by the name of the branch you want to create:

   ```bash
   git branch my-new-branch
   ```

### Pushing the New Branch to a Remote Repository

1. **Add your changes** to the new branch (if any):

   ```bash
   git add .
   ```

2. **Commit your changes**:

   ```bash
   git commit -m "Your commit message"
   ```

3. **Push the new branch to the remote repository** using the `git push` command followed by the name of the remote (usually `origin`) and the branch name:

   ```bash
   git push origin my-new-branch
   ```

### Verifying the New Branch

To verify that your new branch has been created and pushed successfully, you can use the following commands:

1. **List all local branches** to ensure your new branch is created:

   ```bash
   git branch
   ```

2. **List all remote branches** to ensure your new branch is pushed:

   ```bash
   git branch -r
   ```

By following these steps, you can create a new branch in your local Git repository, switch to it, make and commit changes, and push the branch to a remote repository.
