# For machine learning course
## describe
</br> SE、lightweight attention、self-attention、sparse-attention and original U-net </br>

## environment
```bash
git clone https://github.com/1198171877/ml_homework.git
```

</br> run </br>
```bash
conda create -n unet python=3.11
```
```bash
conda activate unet
```
```bash
pip3 install -r requirements.txt
```
## Dataset
</br> put your dataset in datasets,orginize as two-class segmentation task </br>
</br> can also download SOS datasets </br>

## Usages
### examples

```python
python3 Main.py --image_height 256 --image_width 256 --num_classes 2  --dataset_path ./datasets/SOS/palsar --epochs 100 
```


### other options
</br> check utils/parser.py for details</br>

## Visual training process
</br>default "runs" directory will be created automatically,and you can do in terminal</br>
 ```bash 
tensorboard  --port=1937 --logdir runs 
```
</br> check localhost:1937 for detail<br>

## Todo
1. add tensorboard logs read
2. add reducion support in parser.py
3. add more blocks suport
4. finish test.py
5. add docker image
