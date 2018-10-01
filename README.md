# Project Lernende Agenten

## Git

### How to clone
```bash
git clone git clone --recurse-submodules git@github.com:plushvoxel/Project-Lernende-Agenten.git
git checkout -b dev
```

### How to Push
```bash
git fetch origin master
git rebase master
git checkout master
git rebase dev
git push
git checkout dev
```

