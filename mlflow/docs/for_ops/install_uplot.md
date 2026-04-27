## installing YouPlot

ARM supported**
```bash
sudo apt install ruby-dev
sudo gem install youplot
```

## installing youplot on conda (ibex if needed)
- after openning your conda environment
```bash
conda install -c conda-forge ruby compilers
gem install youplot
uplot --version # to check if installation succeeded
```
Another check:

```bash
curl -sL https://git.io/AirPassengers \
| cut -f2,3 -d, \
| uplot line -d, -w 50 -h 15 -t AirPassengers --xlim 1950,1960 --ylim 0,600

```

