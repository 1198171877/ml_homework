# For machine learning course
## environment
</br> run </br>
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
</br>default "runs" directory will be created automatically,and you can do </br>
        ```bash tensorboard  --port=1937 --logdir runs ```
