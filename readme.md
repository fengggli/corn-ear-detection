1. [some notes](lecture-notes.md)
2. [environment](notes-feng.md)

3. Prepare environment 

```
conda create -n tf-cpu pip pillow lxml jupyter matplotlib opencv cython python=3.7 tensorflow=1.14
conda activate tf-cpu
```

4. test images
```
python tests/detect.py --imagepath data/examples/file16frame100.jpg
```

5. test video
```
python tests/detect.py 
```
