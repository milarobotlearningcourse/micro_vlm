

# Running code

```
mila code --cluster mila playground/mini-grp/ --alloc --gres gpu:1 --mem=32G --cpus-per-gpu=6 --partition unkillable
mila code playground/mini-grp --cluster rorqual.alliancecan.ca --alloc --gres gpu:1 --mem=32G --cpus-per-gpu=6 --account=rrg-gberseth --time=2:59:00
```