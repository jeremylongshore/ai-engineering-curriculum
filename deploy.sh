#!/bin/bash

# GitHub Pages Deployment Script
# Run this to deploy your curriculum to GitHub Pages

echo "🚀 Deploying AI Engineering Curriculum to GitHub Pages..."

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Add all files
echo "Adding files..."
git add .

# Commit
echo "Creating commit..."
git commit -m "Deploy AI Engineering Curriculum to GitHub Pages 🚀

- Complete 4-level curriculum
- Interactive web pages for each level
- Teaching manual for educators
- Based on 2024-2025 industry practices

Built with ❤️ for the AI community"

# Add remote (replace with your repository URL)
echo "⚠️  Please update the repository URL in this script!"
echo "Replace 'jeremylongshore' with your GitHub username"
# git remote add origin https://github.com/jeremylongshore/ai-engineering-curriculum.git

# Push to GitHub
# git push -u origin main

echo "
✅ Almost done! To complete deployment:

1. Edit this script and replace 'jeremylongshore' with your GitHub username
2. Create a new repository on GitHub named 'ai-engineering-curriculum'
3. Run this script again
4. Go to Settings > Pages in your GitHub repo
5. Set Source to 'Deploy from a branch'
6. Select 'main' branch and '/ (root)' folder
7. Click Save

Your curriculum will be live at:
https://jeremylongshore.github.io/ai-engineering-curriculum

🌟 Don't forget to:
- Add a custom domain if you have one
- Enable HTTPS
- Add Google Analytics for tracking
- Share with the community!
"